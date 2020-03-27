"""Wrappers for the loss functions

Helper functions that add additional functionalities to loss functions.

"""


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
        ground_truths = args[3]
        # c = ground_truths
        c = ground_truths.size(2)

        perms = get_all_permutations(c)

        min_loss = None
        for perm in perms:
            loss = func(args[0], model_input[perm], args[2], ground_truths)
            if min_loss is None or loss.item() < min_loss.item():
                min_loss = loss

        return min_loss

    return wrapper_perm_invariant


# @perm_invariant
# def test(model_input, model_output, ground_truths):
#     print(model_input)
