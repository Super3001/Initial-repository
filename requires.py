# requires.py
def find_kth_largest(arr, k):
    """
    在数组arr中查找第k%大的元素的值和索引
    """
    n = len(arr)
    k = int(k * n / 100)  # 将k转换为实际索引
    if k < 0 or k >= n:
        return None, None
    
    # 使用快速选择算法查找第k大的元素
    left, right = 0, n - 1
    while left <= right:
        pivot_index = partition(arr, left, right)
        if pivot_index == k:
            return arr[k], k
        elif pivot_index < k:
            left = pivot_index + 1
        else:
            right = pivot_index - 1
    
    return None, None

def partition(arr, left, right):
    """
    将数组arr在[left, right]范围内划分为两个部分，
    左半部分的元素都小于等于arr[pivot_index]，
    右半部分的元素都大于arr[pivot_index]，
    并返回pivot_index。
    """
    pivot_index = left
    pivot_value = arr[right]
    for i in range(left, right):
        if arr[i] <= pivot_value:
            arr[i], arr[pivot_index] = arr[pivot_index], arr[i]
            pivot_index += 1
    arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
    return pivot_index

arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
k = 50  # 查找第50%大的元素
value, index = find_kth_largest(arr, k)
print("第%d%%大的元素是%d，索引是%d" % (k, value, index))
