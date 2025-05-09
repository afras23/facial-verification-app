import tensorflow as tf
from layers import L1Dist

def test_l1_distance():
    l1 = L1Dist()
    a = tf.constant([[1.0, 2.0, 3.0]])
    b = tf.constant([[2.0, 2.0, 4.0]])
    result = l1(a, b).numpy()
    assert all(result == [1.0, 0.0, 1.0]), f"Unexpected output: {result}"

if __name__ == "__main__":
    test_l1_distance()
    print("L1Dist test passed!")
