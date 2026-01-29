import torch

DEBUG = False
_PRINTED_MSGS = set()

LORSUM_DTYPE = torch.float32


def _to_dtype(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == LORSUM_DTYPE:
        return t
    return t.to(dtype=LORSUM_DTYPE)


def safe_inv(matrix, eps=1e-8):
    try:
        matrix_inv = torch.linalg.inv(matrix)
    except torch._C._LinAlgError:
        global _PRINTED_MSGS
        from logging import warning
        warning_msg = "Error inverting matrix!!!"
        if warning_msg not in _PRINTED_MSGS:
            warning(warning_msg)
            _PRINTED_MSGS.add(warning_msg)
        # resort to identity when inverse fails
        matrix_inv = torch.eye(matrix.shape[0], dtype=LORSUM_DTYPE, device=matrix.device).to(matrix)

    return matrix_inv


def _solve(A: torch.Tensor, B: torch.Tensor, inv_free: bool = True, eps: float = 1e-6) -> torch.Tensor:
    if inv_free:
        n = A.shape[-1]
        I = torch.eye(n).to(A)
        L, info = torch.linalg.cholesky_ex(A + I.mul(eps))
        if int(info) == 0:
            return torch.cholesky_solve(B, L)
        # Fallback
        return torch.linalg.solve(A, B)
    
    return torch.linalg.inv(A) @ B


def _solve_sqrt(A: torch.Tensor, B: torch.Tensor, inv_free: bool) -> torch.Tensor:
    return _solve(sqrtm_newton_schulz(A), B, inv_free=inv_free)


# @torch.jit.script
def sqrtm_newton_schulz(A: torch.Tensor, num_iters: int = 6, eps: float = 1e-6) -> torch.Tensor:
    # return naive_sqrtm(A)  # for debugging
    """
    Fast matrix square-root via Newton-Schulz iteration. (By GPT-5)
    Args
    ----
    A : (..., n, n)  Symmetric positive-definite (SPD) or at least
                     with eigenvalues in the open right-half plane.
    num_iters : int  Number of NS iterations (5-8 is usually enough).
    eps : float      Jitter added to the diagonal when A is close to
                     singular to keep things stable.

    Returns
    -------
    (..., n, n)  Matrix square-root of A.
    """

    # Add a tiny ridge if the input might be poorly conditioned
    if eps > 0:
        eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        A = A + eps * eye

    # Frobenius-norm scaling (Higham 1986) – improves convergence
    normA = A.flatten(-2).norm(p=2, dim=-1)  # (...,)
    Y = A / normA[..., None, None].add(eps)      # (..., n, n)
    I = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    I = I.expand_as(Y)
    Z = I.clone()

    for _ in range(num_iters):
        # Newton–Schulz transform
        T = 0.5 * (3.0 * I - Z @ Y)   # (..., n, n)
        Y = Y @ T                     # (..., n, n)
        Z = T @ Z                     # (..., n, n)

    return Y * torch.sqrt(normA)[..., None, None]   # rescale


@torch.no_grad()
def precond_lora_grad_(
    weight_in: torch.Tensor,
    weight_out: torch.Tensor,
    lmbd: float = 0.0,
    lmbd_in: float = 0.0,  # TODO: remove
    lmbd_out: float = 0.0,  # TODO: remove
    inv_free: bool = True,
) -> None:

    # Handle choice of weight decay param
    lmbd_in = max(lmbd_in, lmbd)
    lmbd_out = max(lmbd_out, lmbd)

    eye_r = torch.eye(weight_in.shape[0]).to(weight_in)

    if weight_in.grad is not None:
        weight_in.grad = _solve(
            weight_out.T @ weight_out + eye_r.mul(lmbd_in),
            weight_in.grad,
            inv_free=inv_free
        )
    if weight_out.grad is not None:
        weight_out.grad = _solve(
            weight_in @ weight_in.T + eye_r.mul(lmbd_out),
            weight_out.grad.T,
            inv_free=inv_free
        ).T


# @torch.jit.script
def low_rank_sum(
    factors: list[tuple[torch.Tensor, torch.Tensor]],
    coefficients: list[float],
    start_turn: str = "in",
    num_iters: int = 1,
    lmbd: float = 0.0,
    lmbd_in: float = 0.0,  # TODO: remove
    lmbd_out: float = 0.0,  # TODO: remove
    tol: float = 0.0,  # TODO: test
    inv_free: bool = True,
    is_prox: bool = True,
    normalize_coefficients: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:

    # Handle choice of weight decay param
    lmbd_in = max(lmbd_in, lmbd)
    lmbd_out = max(lmbd_out, lmbd)

    assert len(factors) >= 2, "factors must be a list of at least two tuples"
    assert len(factors) == len(coefficients), "coefficients must match the number of factors"

    # ---- Force fp32 for all computations, then cast outputs back ---- #
    weight_in_orig, weight_out_orig = factors[0]
    weight_in_dtype = weight_in_orig.dtype
    weight_out_dtype = weight_out_orig.dtype
    weight_in_device = weight_in_orig.device

    factors = [(_to_dtype(f_in), _to_dtype(f_out)) for f_in, f_out in factors]

    if num_iters == 0:
        num_iters = 1
        alternating = False
    else:
        alternating = True

    # XXX: questionable addition
    if normalize_coefficients:
        coeff_sum = sum(abs(c) for c in coefficients)
        coefficients = [c / coeff_sum for c in coefficients]
    else:
        coeff_sum = 1.0

    # Pay extra attention to the _t suffix, these are the lookahead weights
    weight_in, weight_out = factors[0]
    weight_in_t, weight_out_t = weight_in.clone(), weight_out.clone()
    eye_r = torch.eye(weight_in_t.shape[0], device=weight_in_device, dtype=weight_in_dtype)

    for t in range(int(2 * num_iters)):
        prev_in  = weight_in_t.clone()
        prev_out = weight_out_t.clone()

        if start_turn == "in" and t % 2 == 0 or start_turn == "out" and t % 2 == 1:
            # ---------- Weight in ---------- #
            sum_in = weight_in.mul(lmbd_in if is_prox else 0.0)
            for coeff, (factor_in, factor_out) in zip(coefficients, factors):
                sum_in.add_((weight_out_t.T @ factor_out) @ factor_in, alpha=coeff)
            weight_in_sol = _solve(weight_out_t.T @ weight_out_t + eye_r.mul(lmbd_in), sum_in, inv_free=inv_free)
            if alternating:
                weight_in_t = weight_in_sol

        else:
            # ---------- Weight out ---------- #
            sum_out = weight_out.mul(lmbd_out if is_prox else 0.0)
            for coeff, (factor_in, factor_out) in zip(coefficients, factors):
                sum_out.add_(factor_out @ (factor_in @ weight_in_t.T), alpha=coeff)
            weight_out_sol = _solve(weight_in_t @ weight_in_t.T + eye_r.mul(lmbd_out), sum_out.T, inv_free=inv_free).T
            if alternating:
                weight_out_t = weight_out_sol

        # ---- Early stop ----
        if tol > 0:
            rel_change = max(
                (weight_in_t - prev_in).norm()  / (prev_in.norm() + 1e-12),
                (weight_out_t - prev_out).norm() / (prev_out.norm() + 1e-12),
            )
            if rel_change < tol:
                break

    if normalize_coefficients:
        weight_in_t, weight_out_t = weight_in_t * coeff_sum ** 0.5, weight_out_t * coeff_sum ** 0.5

    return weight_in_t.to(dtype=weight_in_dtype), weight_out_t.to(dtype=weight_out_dtype)

# @torch.jit.script
def scaled_low_rank_sum(
    factors: list[tuple[torch.Tensor, torch.Tensor]],
    coefficients: list[float],
    metrics: tuple[torch.Tensor, torch.Tensor],
    start_turn: str = "in",
    num_iters: int = 1,
    lmbd: float = 0.0,
    inv_free: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:

    lmbd_in = lmbd_out = lmbd
    assert len(factors) >= 2, "factors must be a list of at least two tuples"
    assert len(factors) == len(coefficients), "coefficients must match the number of factors"
    # assert abs(coefficients[0] - 1.0) < 1e-6, "first coefficient must be 1.0"

    # ---- Force fp32 for all computations, then cast outputs back ---- #
    weight_in, _ = factors[0]
    dtype = weight_in.dtype
    device = weight_in.device

    factors = [(_to_dtype(f_in), _to_dtype(f_out)) for f_in, f_out in factors]
    metrics = (_to_dtype(metrics[0]), _to_dtype(metrics[1]))

    if num_iters == 0:
        num_iters = 1
        alternating = False
    else:
        alternating = True

    # Pay extra attention to the _t suffix, these are the lookahead weights
    weight_in, weight_out = factors[0]
    # input, grad_output_T = factors[1]
    # _, lr = coefficients
    weight_in_t, weight_out_t = weight_in.clone(), weight_out.clone()
    eye_r = torch.eye(weight_in_t.shape[0], device=device, dtype=weight_in_t.dtype)
    metric_in, metric_out = metrics

    # factor_in: (r, n)
    # factor_out: (m, r)
    # metric_in: (n, n) or (n,)
    # metric_out: (m, m) or (m,)

    # Precompute preconditioned and scaled factors
    if metric_in.ndim == 1:
        pfactors = [
            (
                factor_in * metric_in.reciprocal().view(1, -1),
                metric_out.reciprocal().view(-1, 1) * factor_out,
            )
            for factor_in, factor_out in factors[1:]
        ]
    else:
        pfactors = [
            (
                _solve(metric_in, factor_in.T, inv_free=inv_free).T,
                _solve(metric_out, factor_out, inv_free=inv_free),
            )
            for factor_in, factor_out in factors[1:]
        ]
    pfactors = [None] + pfactors  # add dummy for the first factor to match length

    # if metric_in.ndim == 1:
    #     sfactors = [
    #         (
    #             factor_in * metric_in.view(1, -1),
    #             metric_out.view(-1, 1) * factor_out,
    #         )
    #         for factor_in, factor_out in factors
    #     ]
    # else:
    #     sfactors = [
    #         (
    #             factor_in @ metric_in,
    #             metric_out @ factor_out,
    #         )
    #         for factor_in, factor_out in factors
    #     ]

    # --- Main loop --- #
    weight_in_out, weight_out_out =  weight_in_t, weight_out_t
    for t in range(int(2 * num_iters)):
        if start_turn == "in" and t % 2 == 0 or start_turn == "out" and t % 2 == 1:
            # ---------- Weight in ---------- #
            # Compute scaled weight_out_t
            if metric_out.ndim == 1:
                sweight_out_t = metric_out.view(-1, 1) * weight_out_t
            else:
                sweight_out_t = metric_out @ weight_out_t
            # Compute update
            pivot = weight_in.mul(lmbd_in)
            pivot.add_((sweight_out_t.T @ weight_out) @ weight_in, alpha=coefficients[0])
            for j in range(1, len(coefficients)):
                pfactor_in, _ = pfactors[j]
                _, factor_out = factors[j]
                # _, sfactor_out = sfactors[j]
                pivot.add_(
                    (weight_out_t.T @ factor_out) @ pfactor_in,
                    alpha=coefficients[j]
                )
            # Precondition
            weight_in_out = _solve(
                sweight_out_t.T @ weight_out_t + eye_r.mul(lmbd_in),
                pivot,
                inv_free=inv_free
            )
            if alternating:
                weight_in_t = weight_in_out

        else:
            # ---------- Weight out ---------- #
            # Compute scaled weight_in_t
            if metric_in.ndim == 1:
                sweight_in_t = weight_in_t * metric_in.view(1, -1)
            else:
                sweight_in_t = weight_in_t @ metric_in
            # Compute update
            pivot = weight_out.mul(lmbd_out)
            pivot.add_(weight_out @ (weight_in @ sweight_in_t.T), alpha=coefficients[0])
            for j in range(1, len(coefficients)):
                _, pfactor_out = pfactors[j]
                factor_in, _ = factors[j]
                # sfactor_in, _ = sfactors[j]
                pivot.add_(
                    pfactor_out @ (factor_in @ weight_in_t.T),
                    alpha=coefficients[j]
                )
            # Precondition
            weight_out_out = _solve(
                weight_in_t @ sweight_in_t.T + eye_r.mul(lmbd_out),
                pivot.T,
                inv_free=inv_free
            ).T
            weight_out_t = weight_out_out

    return weight_in_out.to(dtype=dtype), weight_out_out.to(dtype=dtype)


# @torch.jit.script
def _scaled_low_rank_sum_debug(
    factors: list[tuple[torch.Tensor, torch.Tensor]],
    coefficients: list[float],
    metrics: tuple[torch.Tensor, torch.Tensor],
    precomputed_metric_inverses: tuple[torch.Tensor, torch.Tensor] | None = None,
    start_turn: str = "in",
    num_iters: int = 1,
    lmbd: float = 0.0,
    min_lmbd: float = 1e-3,  # TODO: make this choice explicitly passed from scaled oplora
    inv_free: bool = True,
    sqrt: bool = False,
    return_grads: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:

    lmbd = max(lmbd, min_lmbd)
    lmbd_in = lmbd_out = lmbd
    assert len(factors) >= 2, "factors must be a list of at least two tuples"
    assert len(factors) == len(coefficients), "coefficients must match the number of factors"

    if num_iters == 0:
        num_iters = 1
        alternating = False
    else:
        alternating = True

    # Pay extra attention to the _t suffix, these are the lookahead weights
    weight_in, weight_out = factors[0]
    # input, grad_output_T = factors[1]
    # _, lr = coefficients
    weight_in_t, weight_out_t = weight_in.clone(), weight_out.clone()
    eye_r = torch.eye(weight_in_t.shape[0]).to(weight_in_t)
    metric_in, metric_out = metrics

    # factor_in: (r, n)
    # factor_out: (m, r)
    # metric_in: (n, n) or (n,)
    # metric_out: (m, m) or (m,)

    # Precompute preconditioned and scaled factors
    if precomputed_metric_inverses is None:
        if metric_in.ndim == 1:
            pfactors = [
                (
                    factor_in * metric_in.view(1, -1).pow(-1),
                    metric_out.view(-1, 1).pow(-1) * factor_out,
                )
                for factor_in, factor_out in factors[1:]
            ]
        else:
            pfactors = [
                (
                    _solve(metric_in, factor_in.T, inv_free=inv_free).T,
                    _solve(metric_out, factor_out, inv_free=inv_free),
                )
                for factor_in, factor_out in factors[1:]
            ]
    else:
        inv_metric_in, inv_metric_out = precomputed_metric_inverses
        pfactors = [
          (
            (inv_metric_in @ factor_in.T).T,
            inv_metric_out @ factor_out,
            )
            for factor_in, factor_out in factors[1:]
        ]
    pfactors = [None] + pfactors  # add dummy for the first factor to match length
    
    if metric_in.ndim == 1:
        sfactors = [
            (
                factor_in * metric_in.view(1, -1),
                metric_out.view(-1, 1) * factor_out,
            )
            for factor_in, factor_out in factors
        ]
    else:
        sfactors = [
            (
                factor_in @ metric_in,
                metric_out @ factor_out,
            )
            for factor_in, factor_out in factors
        ]

    # --- Main loop --- #
    weight_in_out, weight_out_out =  weight_in_t, weight_out_t
    for i in range(int(2 * num_iters)):
        if start_turn == "in" and i % 2 == 0 or start_turn == "out" and i % 2 == 1:
            # ---------- Weight in ---------- #
            # Compute scaled weight_out_t
            if metric_out.ndim == 1:
                sweight_out_t = metric_out.view(-1, 1) * weight_out_t
            else:
                sweight_out_t = metric_out @ weight_out_t
            # Compute update
            pivot = weight_in.mul(lmbd_in) + (sweight_out_t.T @ weight_out) @ weight_in
            grad_in = pivot.clone().mul(0)
            for i in range(1, len(coefficients)):
                pfactor_in, _ = pfactors[i]
                _, sfactor_out = sfactors[i]
                pivot.add_(
                    (weight_out_t.T @ sfactor_out) @ pfactor_in,
                    alpha=coefficients[i]
                )
                grad_in.add_(
                    (weight_out_t.T @ sfactor_out) @ pfactor_in,
                )
            # Precondition
            weight_in_out = _solve(
                sweight_out_t.T @ weight_out_t + eye_r.mul(lmbd_in),
                pivot,
                inv_free=inv_free
            )
            grad_in = _solve(
                sweight_out_t.T @ weight_out_t + eye_r.mul(lmbd_in),
                grad_in,
                inv_free=inv_free
            )
            weight_in_proj = weight_in_out - grad_in
            if alternating:
                weight_in_t = weight_in_out

        else:
            # ---------- Weight out ---------- #
            # Compute scaled weight_in_t
            if metric_in.ndim == 1:
                sweight_in_t = weight_in_t * metric_in.view(1, -1)
            else:
                sweight_in_t = weight_in_t @ metric_in
            # Compute update
            pivot = weight_out.mul(lmbd_out) + weight_out @ (weight_in @ sweight_in_t.T)
            grad_out = pivot.clone().mul(0)
            for i in range(1, len(coefficients)):
                _, pfactor_out = pfactors[i]
                sfactor_in, _ = sfactors[i]
                pivot.add_(
                    pfactor_out @ (sfactor_in @ weight_in_t.T),
                    alpha=coefficients[i]
                )
                grad_out.add_(
                    pfactor_out @ (sfactor_in @ weight_in_t.T),
                )
            # Precondition
            weight_out_out = _solve(
                weight_in_t @ sweight_in_t.T + eye_r.mul(lmbd_out),
                pivot.T,
                inv_free=inv_free
            ).T
            grad_out = _solve(
                weight_in_t @ sweight_in_t.T + eye_r.mul(lmbd_out),
                grad_out.T,
                inv_free=inv_free
            ).T
            weight_out_proj = weight_out_out - grad_out
            weight_out_t = weight_out_out

    if return_grads:
        return weight_in_proj, weight_out_proj, grad_in, grad_out
    else:
        return weight_in_out, weight_out_out


def naive_orthogonalize(G) -> torch.Tensor:
    U, S, Vt = torch.linalg.svd(G, full_matrices=False)
    return U, Vt


def naive_sqrtm(G) -> torch.Tensor:
    U, S, Vt = torch.linalg.svd(G, full_matrices=False)
    return U @ torch.diag(torch.sqrt(S)) @ Vt


def sqrt_gram(X, which="right") -> torch.Tensor:
    """'right' returns sqrt(X^T X), 'left' returns sqrt(XX^T)"""
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    if which == "right":
        return Vt.T @ torch.diag(S) @ Vt
    else:
        return U @ torch.diag(S) @ U.T


# ========================= #
# ===== Bregman stuff ===== #

def cubic_root_newton_step(a, b, c, x0=0, num_iters=5, min_val=-float('inf'), max_val=float('inf'), eps=1e-10, ftol=1e-10):
    """
    Solves the following cubic equation using Newton's method:
    x^3 + a x^2 + b x + c = 0
    """
    assert min_val <= x0 <= max_val

    def f(x):
        return x ** 3 + a * x ** 2 + b * x + c

    def g(x):
        return 3 * x ** 2 + 2 * a * x + b

    x = x0
    for _ in range(num_iters):
        x -= f(x) / (g(x) + eps)  # finds root
        x = min(max(x, min_val), max_val)
        if f(x) < ftol:
            break
    
    return x


def scalar_quartic_shrinkage_(
        weights,
        weights_next,
        metrics=None,
        sigma=0.001,
        alpha=0.01
    ) -> None:
    """
    Applies scalar quartic shrinkage to the LoRA weights in-place.
    The Bregman potential is:
        h(U, V) = (sigma / 2) * (||U||_F^2 + ||V||_F^2) + (alpha / 4) * (||U||_F^2 + ||V||_F^2)^2
    """
    weight_in, weight_out = weights
    weight_in_next, weight_out_next = weights_next
    if metrics is not None:
        metric_in, metric_out = metrics
        if metric_in.ndim == 1:
            norms = (weight_in.square() * metric_in.view(1, -1)).sum().item()
            norms += (weight_out.square() * metric_out.view(-1, 1)).sum().item()
        else:
            norms = torch.trace(weight_in @ metric_in @ weight_in.T).item()
            norms += torch.trace(weight_out.T @ metric_out @ weight_out).item()
    else:
        norms = weight_in.square().sum().item() + weight_out.square().sum().item()
    (a, b, c) = (-sigma, 0.0, -alpha * norms)
    shrink = cubic_root_newton_step(a, b, c, x0=1.0, num_iters=10, min_val=0.1, max_val=10.0)
    delta_in = weight_in_next.sub(weight_in)
    delta_out = weight_out_next.sub(weight_out)
    weight_in.add_(delta_in, alpha=shrink)
    weight_out.add_(delta_out, alpha=shrink)
    return shrink


def gram_quartic_shrinkage_(
        weights,
        weights_next,
        metrics=None,
        gamma=0.01
    ) -> None:
    """
    Applies gram quartic shrinkage to the LoRA weights in-place.
    The Bregman potential is:
        h(U, V) = (gamma / 4) * (||U^T U + V V^T||_F^2)
    """
    weight_in, weight_out = weights
    weight_in_next, weight_out_next = weights_next
    # Let X = [U; V], X_new = [U_new; V_new], and ΔX = X_new - X
    # Compute core = ΔX.T @ ΔX = ΔU.T @ ΔU + ΔV @ ΔV.T
    delta_in = weight_in_next.sub(weight_in)  # (r,n)
    delta_out = weight_out_next.sub(weight_out)  # (m,r)
    if metrics is not None:
        metric_in, metric_out = metrics
        if metric_in.ndim == 1:
            core = (delta_in * metric_in.view(1, -1)) @ delta_in.T + delta_out.T @ (metric_out.view(-1, 1) * delta_out)  # (r,r)
        else:
            core = delta_in @ metric_in @ delta_in.T + delta_out.T @ metric_out @ delta_out  # (r,r)
    else:
        core = delta_in @ delta_in.T + delta_out.T @ delta_out  # (r,r)
    # Compute eigs: core = P @ diag(eigs) @ P.T
    eigs, P = torch.linalg.eigh(core)
    # Compute shrinkages: s_i = 1.0 / sqrt(1.0 + gamma * eigs)
    s = torch.rsqrt(1.0 + gamma * eigs.clamp_min(0.0))
    # Apply shrinkage: B = P @ diag(s) @ P.T
    B = P @ torch.diag(s) @ P.T
    # Compute updates: ΔX_tilde = ΔX @ B
    weight_in.add_(B.T @ delta_in)
    weight_out.add_(delta_out @ B)
    return B.norm().item()
