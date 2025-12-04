import numpy as np


# 实现与门
# def ADD(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     res = w1 * x1 + w2 * x2
#     if res <= theta:
#         return 0
#     else:
#         return 1

def ADD(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    # 直接用矩阵运算的形式计算结果
    res = w @ x + b
    if res <= 0:
        return 0
    else:
        return 1


# 与非门
def NADD(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    res = w @ x + b
    if res <= 0:
        return 0
    else:
        return 1


# 或门
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    res = w @ x + b
    if res <= 0:
        return 0
    else:
        return 1


# 异或门
def XOR(x1, x2):
    s1 = NADD(x1, x2)
    s2 = OR(x1, x2)
    y = ADD(s1, s2)
    return y


# 测试
# print(ADD(0, 0))
# print(ADD(0, 1))
# print(ADD(1, 0))
# print(ADD(1, 1))

# print(NADD(0, 0))
# print(NADD(0, 1))
# print(NADD(1, 0))
# print(NADD(1, 1))

# print(OR(0, 0))
# print(OR(0, 1))
# print(OR(1, 0))
# print(OR(1, 1))

print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))
