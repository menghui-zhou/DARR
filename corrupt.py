import numpy as np
import torch

def default_corrupt(trainset, ratio, seed):
    """Corrupt labels in trainset."""
    np.random.seed(seed)
    train_labels = np.asarray(trainset.targets)
    num_classes = np.max(train_labels) + 1
    n_train = len(train_labels)
    n_rand = int(len(trainset.data) * ratio)
    randomize_indices = np.random.choice(range(n_train), size=n_rand, replace=False)
    train_labels[randomize_indices] = np.random.choice(np.arange(num_classes), size=n_rand, replace=True)
    trainset.targets = torch.tensor(train_labels).int()
    return trainset

def shift_corrupt(trainset, ratio, seed):
    """Corrupt labels in trainset by cyclically shifting a portion of them."""
    np.random.seed(seed)
    train_labels = np.asarray(trainset.targets)
    num_classes = np.max(train_labels) + 1
    n_train = len(train_labels)
    n_shift = int(n_train * ratio)
    shift_indices = np.random.choice(range(n_train), size=n_shift, replace=False)
    train_labels[shift_indices] = (train_labels[shift_indices] + 1) % num_classes
    trainset.targets = torch.tensor(train_labels).int()
    return trainset

def cyclic_corrupt(trainset, ratio, seed):
    """Randomly select a portion of labels and apply (label + 1) % num_classes."""
    assert 0 <= ratio <= 1., 'ratio is bounded between 0 and 1'
    np.random.seed(seed)
    train_labels = np.asarray(trainset.targets)
    num_classes = np.max(train_labels) + 1
    n_train = len(train_labels)
    n_shift = int(n_train * ratio)
    shift_indices = np.random.choice(n_train, size=n_shift, replace=False)
    train_labels[shift_indices] = (train_labels[shift_indices] + 1) % num_classes
    trainset.targets = torch.tensor(train_labels).int()
    return trainset

def asymmetric_noise(trainset, ratio, seed):
    assert 0 <= ratio <= 1., 'ratio is bounded between 0 and 1'
    np.random.seed(seed)
    train_labels = np.array(trainset.targets)
    train_labels_gt = train_labels.copy()
    for i in range(trainset.num_classes):
        indices = np.where(train_labels == i)[0]
        np.random.shuffle(indices)
        for j, idx in enumerate(indices):
            if j < ratio * len(indices):
                if i == 9:
                    train_labels[idx] = 1  # truck -> automobile
                elif i == 2:
                    train_labels[idx] = 0  # bird -> airplane
                elif i == 3:
                    train_labels[idx] = 5  # cat -> dog
                elif i == 5:
                    train_labels[idx] = 3  # dog -> cat
                elif i == 4:
                    train_labels[idx] = 7  # deer -> horse
    trainset.targets = torch.tensor(train_labels).int()
    return trainset

def noisify_pairflip(trainset, noise, seed=None):
    """Apply pair-flip noise to the labels."""
    y_train = np.array(trainset.targets)
    nb_classes = np.unique(trainset.targets).size
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=seed)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    trainset.targets = torch.tensor(y_train)
    return trainset

def noisify_multiclass_symmetric(trainset, noise, seed=10):
    """Apply symmetric (uniform) noise to the labels."""
    y_train = np.array(trainset.targets)
    nb_classes = np.unique(y_train).size
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        P[0, 0] = 1. - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1. - n
        P[nb_classes - 1, nb_classes - 1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=seed)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    trainset.targets = torch.tensor(y_train)
    return trainset

def multiclass_noisify(y, P, random_state):
    """Flip classes according to transition probability matrix P."""
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]
    assert np.allclose(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()
    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)
    for idx in np.arange(m):
        i = y[idx]
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]
    return new_y
