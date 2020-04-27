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

    def wrapper_perm_invariant(*args, **kwargs):
        model_input = args[1]
        model_output = args[2]
        ground_truths = args[3]
        s = args[0].s
        b = model_input.size(0)

        if (args[0].input_dimensions == 'BN(2M)' and
           args[0].output_dimensions == 'BN(S2M)'):
            n = model_input.size(1)

            model_input = model_input.view(b, n, 2, -1).permute(0, 2, 1, 3)
            model_output = model_output.view(b, n, s, 2, -1).permute(
                0, 2, 3, 1, 4)
            ground_truths = ground_truths.view(b, n, s, 2, -1).permute(
                0, 2, 3, 1, 4)

        # disable permutation invariant training
        if args[0].no_pit:
            loss = func(args[0], model_input, model_output, ground_truths)
            return loss.mean()

        else:
            perms = get_all_permutations(s)
            losses = torch.zeros(b, len(perms))
            for i, perm in enumerate(perms):
                loss = func(args[0], model_input, model_output[:, perm],
                            ground_truths)
                losses[:, i] = loss

            return losses.min(axis=-1).values.mean()

    wrapper_perm_invariant.unwrapped = func

    return wrapper_perm_invariant


def perm_invariant_nomask(func):
    """Decorator that enables permutational-invariant training

    Source: Permutation invariant training of deep models
            for speaker-independent multi-talker speech separation

    """

    def wrapper_perm_invariant(*args, **kwargs):
        prediction = args[1]
        ground_truths = args[2]

        s = args[0].s

        if (args[0].input_dimensions == 'BN(2M)' and
           args[0].output_dimensions == 'BN(S2M)'):
            b = prediction.size(0)
            n = prediction.size(1)

            prediction = prediction.view(b, n, s, 2, -1).permute(
                0, 2, 3, 1, 4)
            ground_truths = ground_truths.view(b, n, s, 2, -1).permute(
                0, 2, 3, 1, 4)
        else:
            b = model_input.size(0)

        perms = get_all_permutations(s)
        losses = torch.zeros(b, len(perms))
        for i, perm in enumerate(perms):
            loss = func(args[0], prediction[:, :, perm], ground_truths)
            losses[:, i] = loss

        return losses.min(axis=-1).values.mean()

    # wrapper_perm_invariant.unwrapped = func

    return wrapper_perm_invariant
