import numpy as np

def fuzzy_union_or(A, B, operator='max'):
    if len(A) != len(B):
        raise ValueError("Fuzzy sets must have same length")

    if operator == 'max':
        return np.maximum(A, B)
    else:
        raise NotImplementedError("Only max operator supported")

def fuzzy_intersection_and(A, B, operator='min'):
    if len(A) != len(B):
        raise ValueError("Fuzzy sets must have same length")

    if operator == 'min':
        return np.minimum(A, B)
    else:
        raise NotImplementedError("Only min operator supported")

def fuzzy_complement_not(A):
    return 1 - A


# Universe of discourse
U = np.array([1, 2, 3, 4, 5])
print(f"Universe of Discourse (U): {U}\n")

# Fuzzy sets
A = np.array([1.0, 0.8, 0.4, 0.1, 0.0])
B = np.array([0.0, 0.1, 0.3, 0.7, 1.0])

print("Original Sets")
print(f"Fuzzy Set A: {A}")
print(f"Fuzzy Set B: {B}\n")

# Fuzzy Union
A_OR_B = fuzzy_union_or(A, B)
print("Fuzzy UNION (A OR B)")
print(A_OR_B, "\n")

# Fuzzy Intersection
A_AND_B = fuzzy_intersection_and(A, B)
print("Fuzzy INTERSECTION (A AND B)")
print(A_AND_B, "\n")

# Fuzzy Complement
NOT_A = fuzzy_complement_not(A)
print("Fuzzy COMPLEMENT (NOT A)")
print(NOT_A, "\n")

NOT_B = fuzzy_complement_not(B)
print("Fuzzy COMPLEMENT (NOT B)")
print(NOT_B)
