"""Wrappers for the loss functions

Helper functions that add additional functionalities to loss functions.

"""


import torch


def get_all_permutations(n):
    if n <= 1:
        return [[0]]

    all_perms = []
    for perm in get_all_permutations(n - 1):
        for i in range(len(perm) + 1):
            all_perms.append(perm[:i] + [n - 1] + perm[i:])

    return all_perms


def perm_invariant(func):
    """Decorator that enables permutational-invariant training

    Source: Permutation invariant training of deep models
            for speaker-independent multi-talker speech separation

    """

    def wrapper_perm_invariant(*args):
        model_input = args[1]
        model_output = args[2]
        ground_truths = args[3]

        b = ground_truths.size(0)
        c = ground_truths.size(1)

        perms = get_all_permutations(c)

        losses = torch.zeros(b, len(perms))
        for i, perm in enumerate(perms):
            loss = func(args[0], model_input, model_output[:, perm],
                        ground_truths)
            losses[:, i] = loss

        return losses.min(axis=-1).values.mean()

    return wrapper_perm_invariant


# @perm_invariant
# def test(model_input, model_output, ground_truths):
#     print(model_input)
