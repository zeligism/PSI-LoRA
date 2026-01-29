from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments, Seq2SeqTrainingArguments


@dataclass
class OPLoRATrainingArguments(TrainingArguments):
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA rank r"}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "dropout rate for LoRA modules"}
    )
    target_modules: Optional[str] = field(
        default=None, metadata={"help": "which modules to add LoRA layer to"}
    )
    use_lora: bool = field(
        default=False, metadata={"help": "whether to finetune using LoRA"}
    )
    use_precond_lora: bool = field(
        default=False, metadata={"help": "whether to finetune using Precond. LoRA.  `use_lora` must also be true."}
    )
    use_svdlora: bool = field(
        default=False, metadata={"help": "whether to finetune using SVDLoRA"}
    )
    use_svdlora_fista: bool = field(
        default=False, metadata={"help": "whether to finetune using SVDLoRA with FISTA. `use_svdlora` must also be true."}
    )
    use_oplora: bool = field(
        default=False, metadata={"help": "whether to finetune using OPLoRA"}
    )
    use_oplora_proj: bool = field(
        default=False, metadata={"help": "whether to finetune using OPLoRA Proj. `use_oplora` must also be true."}
    )
    use_oplora_fista: bool = field(
        default=False, metadata={"help": "whether to finetune using OPLoRA FISTA. `use_oplora` must also be true."}
    )
    use_oplora_scaled: bool = field(
        default=False, metadata={"help": "whether to finetune using Scaled OPLoRA. `use_oplora` must also be true."}
    )
    oplora_lmbd: float = field(
        default=1e-2, metadata={"help": "OPLoRA's lmbd"}
    )
    oplora_lookahead_iters: int = field(
        default=1, metadata={"help": "OPLoRA's lookahead_iters (>= 0)"}
    )
    oplora_damping: float = field(
        default=0.1, metadata={"help": "OPLoRA's damping factor for scaled OPLoRA"}
    )
    oplora_beta1: float = field(
        default=0.9, metadata={"help": "OPLoRA's beta1 for scaled OPLoRA"}
    )
    oplora_beta2: float = field(
        default=0.999, metadata={"help": "OPLoRA's beta2 for scaled OPLoRA"}
    )
    oplora_kfac_metric: bool = field(
        default=False, metadata={"help": "Whether to use KFAC metric for scaled OPLoRA"}
    )
    oplora_metric_power: float = field(
        default=0.5, metadata={"help": "Fractional power for scaled OPLoRA metric (0 to 1)"}
    )
    learning_rate_for_non_lora: float = field(
        default=5e-5, metadata={"help": "Learning rate for non-LoRA parameters (used with scaled OPLoRA)"}
    )
    momentum: float = field(
        default=0.9, metadata={"help": "SGD's momentum"}
    )


@dataclass
class OPLoRASeq2SeqTrainingArguments(Seq2SeqTrainingArguments, OPLoRATrainingArguments):
    pass