import pytest
import torch as th
import numpy as np
import time

from syft.frameworks.torch.mpc.fss import DPF, DIF, n

# Tested in Rustfss/test/
# @pytest.mark.parametrize("op", ["eq", "le"])
# def test_fss_class(op):
#     class_ = {"eq": DPF, "le": DIF}[op]
#     th_op = {"eq": np.equal, "le": np.less_equal}[op]
#     gather_op = {"eq": "__add__", "le": "__add__"}[op]

#     # single value
#     primitive = class_.keygen(n_values=1)
#     alpha, s_00, s_01, *CW = primitive
#     mask = np.random.randint(0, 2 ** n, alpha.shape, dtype=alpha.dtype)  # IID in int32
#     k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

#     x = np.array([0])
#     x_masked = x + k0[0] + k1[0]
#     y0 = class_.eval(0, x_masked, *k0[1:])
#     y1 = class_.eval(1, x_masked, *k1[1:])

#     assert (getattr(y0, gather_op)(y1) == th_op(x, 0)).all()

#     # 1D tensor
#     primitive = class_.keygen(n_values=3)
#     alpha, s_00, s_01, *CW = primitive
#     mask = np.random.randint(0, 2 ** n, alpha.shape, dtype=alpha.dtype)
#     k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

#     x = np.array([0, 2, -2])
#     x_masked = x + k0[0] + k1[0]
#     y0 = class_.eval(0, x_masked, *k0[1:])
#     y1 = class_.eval(1, x_masked, *k1[1:])

#     assert (getattr(y0, gather_op)(y1) == th_op(x, 0)).all()

#     # 2D tensor
#     primitive = class_.keygen(n_values=4)
#     alpha, s_00, s_01, *CW = primitive
#     mask = np.random.randint(0, 2 ** n, alpha.shape, dtype=alpha.dtype)
#     k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

#     x = np.array([[0, 2], [-2, 0]])
#     x_masked = x + k0[0].reshape(x.shape) + k1[0].reshape(x.shape)
#     y0 = class_.eval(0, x_masked, *k0[1:])
#     y1 = class_.eval(1, x_masked, *k1[1:])

#     assert (getattr(y0, gather_op)(y1) == th_op(x, 0)).all()

#     # 3D tensor
#     primitive = class_.keygen(n_values=8)
#     alpha, s_00, s_01, *CW = primitive
#     mask = np.random.randint(0, 2 ** n, alpha.shape, dtype=alpha.dtype)
#     k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

#     x = np.array([[[0, 2], [-2, 0]], [[0, 2], [-2, 0]]])
#     x_masked = x + k0[0].reshape(x.shape) + k1[0].reshape(x.shape)
#     y0 = class_.eval(0, x_masked, *k0[1:])
#     y1 = class_.eval(1, x_masked, *k1[1:])

#     assert (getattr(y0, gather_op)(y1) == th_op(x, 0)).all()


@pytest.mark.parametrize("op", ["eq", "le"])
def torch_to_numpy(op):
    class_ = {"eq": DPF, "le": DIF}[op]
    th_op = {"eq": th.eq, "le": th.le}[op]
    gather_op = {"eq": "__add__", "le": "__add__"}[op]

    # 1D tensor
    primitive = class_.keygen(n_values=3)
    alpha, s_00, s_01, *CW = primitive
    mask = np.random.randint(0, 2 ** n, alpha.shape, dtype=alpha.dtype)
    k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

    x = th.IntTensor([0, 2, -2])
    np_x = x.numpy()
    x_masked = np_x + k0[0] + k1[0]
    y0 = class_.eval(0, x_masked, *k0[1:])
    y1 = class_.eval(1, x_masked, *k1[1:])

    np_result = getattr(y0, gather_op)(y1)
    th_result = th.native_tensor(np_result)

    assert (th_result == th_op(x, 0)).all()


@pytest.mark.parametrize("op", ["eq"])
# @pytest.mark.parametrize("op", ["eq", "le"])


def test_using_crypto_store(workers, op):
    alice, bob, me = workers["alice"], workers["bob"], workers["me"]
    class_ = {"eq": DPF, "le": DIF}[op]
    th_op = {"eq": th.eq, "le": th.le}[op]
    gather_op = {"eq": "__add__", "le": "__add__"}[op]
    primitive = {"eq": "fss_eq", "le": "fss_comp"}[op]

    me.crypto_store.provide_primitives(primitive, [alice, bob], n_instances=6)
    keys_a = alice.crypto_store.get_keys(primitive, 3, remove=True)
    keys_b = bob.crypto_store.get_keys(primitive, 3, remove=True)

    print(f"Got keys {keys_a} {keys_b}")

    alpha_a = np.frombuffer(np.ascontiguousarray(keys_a[:, 0:4]), dtype=np.uint32).astype(np.uint64)
    alpha_b = np.frombuffer(np.ascontiguousarray(keys_b[:, 0:4]), dtype=np.uint32).astype(np.uint64)

    print(f"And alpha {alpha_a + alpha_b}")

    x = th.IntTensor([0, 2, -2])
    np_x = x.numpy()
    x_masked = (np_x + alpha_a + alpha_b).astype(np.uint64)

    print("Time to eval!")
    y0 = class_.eval(0, x_masked, keys_a)
    print(f"Evaluating the rest")
    y1 = class_.eval(1, x_masked, keys_b)

    np_result = getattr(y0, gather_op)(y1)
    print(f"np result {np_result}")
    th_result = th.native_tensor(np_result)
    print(f"th result {th_result}")
    print(f"Should be {th_op(x, 0)}")

    assert (th_result == th_op(x, 0)).all()


@pytest.mark.parametrize("op", ["eq"])
def test_fat_keygen(workers, op):
    n_instances = 500_000

    alice, bob, me = workers["alice"], workers["bob"], workers["me"]
    class_ = {"eq": DPF, "le": DIF}[op]
    th_op = {"eq": th.eq, "le": th.le}[op]
    gather_op = {"eq": "__add__", "le": "__add__"}[op]
    primitive = {"eq": "fss_eq", "le": "fss_comp"}[op]

    t = time.time()

    me.crypto_store.provide_primitives(primitive, [alice, bob], n_instances=n_instances)
    keys_a = alice.crypto_store.get_keys(primitive, n_instances, remove=True)
    keys_b = bob.crypto_store.get_keys(primitive, n_instances, remove=True)

    print(f"Generated and got {n_instances} primitives in {time.time() - t}.")

    assert False
