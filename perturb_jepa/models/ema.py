from __future__ import annotations

import copy

import torch
from torch import nn


def make_ema_teacher(module: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(module)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema_teacher(student: nn.Module, teacher: nn.Module, *, decay: float = 0.996) -> None:
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters(), strict=True):
        teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
    for student_buffer, teacher_buffer in zip(student.buffers(), teacher.buffers(), strict=True):
        teacher_buffer.copy_(student_buffer)
