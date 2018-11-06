# -*- coding: utf-8 -*-
class Solution_85:
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        res = 0
        height = []
        for i, row in enumerate(matrix):
            height = height[:len(row)] if len(height) > len(row) else \
                [0 for i in range(len(row))]
            for j, e in enumerate(row):
                # height[j] = matrix[i][j] == '0'? 0: (1 + height[j]);
                height[j] = 0 if e == '0' else 1 + height[j]
            res = max(res, self.largestRectangleArea(height))
        return res

    def largestRectangleArea(self, height):
        res = 0
        s = []
        height.append(0)
        i = 0
        # for i, h in enumerate(height):
        while i < len(height):
            if len(s) == 0 or height[s[-1]] <= height[i]:
                s.append(i)
            else:
                tmp = s[-1]
                s = s[:-1]
                width = i if len(s) == 0 else i - s[-1] - 1
                res = max(res, height[tmp] * width)
                i -= 1
            i += 1
        return res

class Solution_84:
    def largestRectangleArea(self, heights):
        """
        单调栈
        :type heights: List[int]
        :rtype: int
        """
        res = 0
        stack = []
        heights.append(0)

        i = 0
        while i < len(heights):
            if len(stack) == 0 or heights[stack[-1]] <= heights[i]:
                stack.append(i)
                i += 1
            else:
                tmp = stack[-1]
                stack = stack[:-1]
                width = i if len(stack) == 0 else i - stack[-1] - 1
                res = max(res, heights[tmp] * width)

        return res


class Solution_42:
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        res = 0
        mx = 0
        n = len(height)
        dp = [0 for i in range(n)]

        for i in range(n):
            dp[i] = mx
            mx = max(mx, height[i])

        mx = 0
        for i in range(n - 1, -1, -1):
            dp[i] = min(dp[i], mx)
            mx = max(mx, height[i])
            res += (dp[i] - height[i]) if dp[i] > height[i] else 0

        return res


class Solution_46:
    def permute(self, nums):
        """
        permutations
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        self.permute_internal(nums, 0, res)
        return res

    def permute_internal(self, num_, start, res):
        if start == len(num_):
            res.append(num_)

        i = start
        nums = copy.deepcopy(num_)
        while i < len(nums):
            nums[start], nums[i] = nums[i], nums[start]
            self.permute_internal(nums, start + 1, res)
            nums[start], nums[i] = nums[i], nums[start]
            i += 1


class Solution:
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # res = set() TLE
        res = []
        self.permute_internal(nums, 0, res)
        # res = list(res)
        # res = list(map(lambda x: list(x), res)) TLE
        return res

    def permute_internal(self, nums, start, res):
        if (start >= len(nums)):
            res.append(nums)

        visited = set()
        i = start
        nums_ = copy.deepcopy(nums)
        while i < len(nums):
            if start == i or (nums_[start] != nums_[i] and nums[i] not in visited):
                nums_[start], nums_[i] = nums_[i], nums_[start]
                self.permute_internal(nums_, start + 1, res)
                nums_[start], nums_[i] = nums_[i], nums_[start]
                visited.add(nums[i])
            i += 1

    # below is a TLE method
    def permute_internal1(self, nums, start, res):
        if (start >= len(nums)):
            res.add(tuple(nums))

        i = start
        nums_ = copy.deepcopy(nums)
        while i < len(nums):
            nums_[start], nums_[i] = nums_[i], nums_[start]
            self.permute_internal(nums_, start + 1, res)
            nums_[start], nums_[i] = nums_[i], nums_[start]
            i += 1


class Solution_31:
    def nextPermutation(self, nums):
        """
        next permutation
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        cur, ind1, ind2, exist = float('-inf'), -1, -1, False
        for i in range(len(nums) - 1, -1, -1):
            if cur > nums[i]:
                exist = True
                ind1 = i
                break
            cur = nums[i]

        if not exist:
            # reversed(nums)
            nums.reverse()
            return

        mi = float('inf')
        for j in range(ind1 + 1, len(nums)):
            if nums[j] > nums[ind1] and nums[j] <= mi:  # find the last largest number
                mi = nums[j]
                ind2 = j

        # print(ind1, ind2)
        nums[ind1], nums[ind2] = nums[ind2], nums[ind1]
        nums[ind1 + 1:len(nums)] = nums[ind1 + 1:len(nums)][::-1]
        # print(nums)


class Solution_39:
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        all_sol = []
        self.comb_sum_internal(candidates, 0, [], all_sol, target)
        return all_sol

    def comb_sum_internal(self, candidates, ind, cur_sol, all_sol, target):
        if target == 0:
            all_sol.append(cur_sol)

        if target < 0:
            return

        i = ind
        while i < len(candidates):
            cur_sol_cp = copy.deepcopy(cur_sol)
            cur_sol_cp.append(candidates[i])
            self.comb_sum_internal(candidates, i, cur_sol_cp, all_sol,
                target - candidates[i])
            # cur_sol_cp = cur_sol_cp[:-1]
            i += 1


class Solution_40:
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        # all_sol = set()
        all_sol = []
        candidates = sorted(candidates)
        self.comb_sum_internal(candidates, 0, [], all_sol, target)
        all_sol = set(map(lambda x: tuple(x), all_sol))
        return list(all_sol)

    def comb_sum_internal(self, candidates, ind, cur_sol, all_sol, target):
        if target == 0:
            # all_sol.add(tuple(sorted(cur_sol)))
            all_sol.append(cur_sol)

        i = ind
        while i < len(candidates):
            cur_sol_cp = copy.deepcopy(cur_sol)
            cur_sol_cp.append(candidates[i])
            if target < candidates[i]:
                return
            self.comb_sum_internal(candidates, i + 1, cur_sol_cp, all_sol,
                                   target - candidates[i])
            # cur_sol_cp = cur_sol_cp[:-1]
            i += 1

class Solution_932:
    def beautifulArray(self, N):
        memo = {1: [1]}
        def f(N):
            if N not in memo:
                odds = f((N+1)//2)
                evens = f(N//2)
                memo[N] = [2*x-1 for x in odds] + [2*x for x in evens]
            return memo[N]
        return f(N)

class Solution_931:
    def minFallingPathSum(self, A):
        """
        :type A: List[List[int]]
        :rtype: int
        """
        while len(A) >= 2:
            row = A.pop()
            for i in range(len(row)):
                A[-1][i] += min(row[max(0, i-1): min(len(row), i+2)])
        return min(A[0])


class Solution_310:
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        if n == 1:
            return [0]

        res = []
        adj = [set() for i in range(n)]
        q = []

        # init adj set
        for edge in edges:
            adj[edge[0]].add(edge[1])
            adj[edge[1]].add(edge[0])

        # init queue
        for i in range(n):
            if len(adj[i]) == 1:
                q.append(i)

        while n > 2:
            size = len(q)
            n -= size
            for i in range(size):
                t = q[0]
                q = q[1:]
                for a in adj[t]:
                    adj[a].remove(t)
                    if len(adj[a]) == 1:
                        q.append(a)

        while (len(q) > 0):
            res.append(q[0])
            q = q[1:]

        return res


class Solution_207:
    def canFinish(self, numCourses, prerequisites):
        """
        course schedule
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        graph = [[] for i in range(numCourses)]
        visit = [0 for i in range(numCourses)]

        for prer in prerequisites:
            graph[prer[1]].append(prer[0])

        for i in range(numCourses):
            if not self.canFinish_internal(graph, visit, i):
                return False
        return True

    def canFinish_internal(self, graph, visit, i):
        if visit[i] == -1:
            return False
        if visit[i] == 1:
            return True
        visit[i] = -1
        for a in graph[i]:
            if not self.canFinish_internal(graph, visit, a):
                return False
        visit[i] = 1
        return True


class Solution_preorder_recursive:
    """
    @param root: A Tree
    @return: Preorder in ArrayList which contains node values.
    """

    def preorderTraversal(self, root):
        # write your code here
        res = []
        self.preorder_internal(root, res)
        return res

    def preorder_internal(self, root, res):
        if not root:
            return
        res.append(root.val)
        self.preorder_internal(root.left, res)
        self.preorder_internal(root.right, res)


class Solution_preorder_iterative:
    """
    @param root: A Tree
    @return: Preorder in ArrayList which contains node values.
    """

    def preorderTraversal(self, root):
        # write your code here
        res = []
        s = []  # the stack

        if root:
            s.append(root)

        while len(s) > 0:
            node = s.pop()
            if not node:
                continue
            res.append(node.val)
            s.append(node.right)
            s.append(node.left)
        return res


import copy


class Solution_binary_tree_paths:
    """
    @param root: the root of the binary tree
    @return: all root-to-leaf paths
    """

    def binaryTreePaths(self, root):
        # write your code here
        res = []
        s = []  # stack
        if root:
            s.append((root, []))

        while len(s) > 0:
            node, cur_path = s[0]
            s = s[1:]
            cur_path_cp = copy.deepcopy(cur_path)
            cur_path_cp.append(str(node.val))
            if not node.left and not node.right:
                res.append(cur_path_cp)
            if node.left:
                s.append((node.left, cur_path_cp))
            if node.right:
                s.append((node.right, cur_path_cp))

        return list(map(lambda x: "->".join(x), res))

class Solution_remove_dep:
    """
    @param: nums: An ineger array
    @return: An integer
    """
    def removeDuplicates(self, nums):
        # write your code here
        pt = 1
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                nums[pt] = nums[i]
                pt += 1
        return pt


class Solution_two_sum:
    """
    @param numbers: An array of Integer
    @param target: target = numbers[index1] + numbers[index2]
    @return: [index1, index2] (index1 < index2)
    """

    def twoSum(self, numbers, target):
        # write your code here
        ele2ind = {}
        for i, e in enumerate(numbers):
            if e in ele2ind:
                ele2ind[e].append(i)
            else:
                ele2ind[e] = [i]

        for i, inds_i in ele2ind.items():
            for ind_i in inds_i:
                if target - i in ele2ind:
                    for ind_j in ele2ind[target - i]:
                        if ind_i != ind_j:
                            return [min(ind_i, ind_j), max(ind_i, ind_j)]

class Solution_reserse_list:
    """
    @param head: n
    @return: The new head of reversed linked list.
    """
    def reverse(self, head):
        # write your code here
        p = ListNode(-1)
        pivot = p
        # pivot
        ptr = head
        while ptr:
            pnext = p.next
            ptrnext = ptr.next
            p.next = ptr
            ptr.next = pnext
            ptr = ptrnext
        return pivot.next


class Solution_insert_cyclic_list:
    """
    @param: node: a list node in the list
    @param: x: An integer
    @return: the inserted new list node
    """

    def insert(self, node, x):
        # write your code here
        pt = node
        ret_node = ListNode(x)
        if not pt:
            ret_node.next = ret_node
            return ret_node

        first_time = True
        while first_time or pt != node:
            if pt.val < pt.next.val and (x >= pt.val and x <= pt.next.val):
                break
            if pt.val == pt.next.val and x == pt.val:
                break
            if pt.val > pt.next.val and (x >= pt.val or x <= pt.next.val):
                break
            pt = pt.next
            first_time = False

        # insert node with val x between pt and pt.next
        pt_next = pt.next
        pt.next = ret_node
        ret_node.next = pt_next

        return ret_node


class Solution_lowest_common_ancester:
    """
    @param: root: The root of the tree
    @param: A: node in the tree
    @param: B: node in the tree
    @return: The lowest common ancestor of A and B
    """

    def lowestCommonAncestorII(self, root, A, B):
        # write your code here
        return self.lca_internal(root, A, B)

    def lca_internal(self, root, A, B):
        if not root or root == A or root == B:
            return root
        left = self.lca_internal(root.left, A, B)
        right = self.lca_internal(root.right, A, B)
        if left and right:
            return root
        return left if left else right

'''cpp
class Solution {
public:
    /*
     * @param num: An integer
     * @return: An integer
     */
    int countOnes(int num) {
        // write your code here
        int count = 0;
        while(num != 0){
            num = num & (num - 1);
            count++;
        }
        return count;
    }
};

'''


class Solution_add_binary:
    """
    @param a: a number
    @param b: a number
    @return: the result
    """

    def addBinary(self, a, b):
        # write your code here
        res = []
        len_a, len_b = len(a), len(b)
        carry = 0

        for i in range(max(len(a), len(b))):
            ia = -1 - i + len(a)
            ib = -1 - i + len(b)
            d_a = a[ia] if ia < len(a) and ia >= 0 else 0
            d_b = b[ib] if ib < len(b) and ib >= 0 else 0
            d_res = int(d_a) + int(d_b) + carry
            carry = d_res >= 2
            d = d_res % 2
            res.append(str(d))
        if carry:
            res.append('1')
        res = res[::-1]
        return ''.join(res)


class Solution_remove_nth_node_from_end_list:
    """
    @param head: The first node of linked list.
    @param n: An integer
    @return: The head of linked list.
    """

    def removeNthFromEnd(self, head, n):
        # write your code here
        pivot = ListNode(-1)
        pivot.next = head
        pt1 = pivot
        pt2 = pivot

        while n:
            pt1 = pt1.next
            n -= 1

        while pt1.next:
            pt1 = pt1.next
            pt2 = pt2.next

        # now remove pt2.next
        # pt2_next = pt2.next
        # pt2_next_next = pt2_next.next
        pt2.next = pt2.next.next

        return pivot.next


class Solution_merge_two_sorted_list:
    """
    @param l1: ListNode l1 is the head of the linked list
    @param l2: ListNode l2 is the head of the linked list
    @return: ListNode head of linked list
    """

    def mergeTwoLists(self, l1, l2):
        # write your code here
        pivot = ListNode(-1)
        pt = pivot

        while l1 or l2:
            if (l1 and l2 and l1.val <= l2.val) or not l2:
                pt.next = ListNode(l1.val)
                pt = pt.next
                l1 = l1.next
                continue
            if (l1 and l2 and l2.val <= l1.val) or not l1:
                pt.next = ListNode(l2.val)
                pt = pt.next
                l2 = l2.next
        return pivot.next


class Solution_max_depth_of_tree:
    """
    @param root: The root of binary tree.
    @return: An integer
    """

    def maxDepth(self, root):
        # write your code here
        s = []
        max_depth = 0
        if root:
            s.append((root, 1))

        while len(s) > 0:
            node, cur_depth = s[0]
            s = s[1:]
            max_depth = max(max_depth, cur_depth)
            if node.left:
                s.append((node.left, cur_depth + 1))
            if node.right:
                s.append((node.right, cur_depth + 1))

        return max_depth


class Solution_is_balanced_tree:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """

    def isBalanced(self, root):
        # write your code here
        h, b = self.is_balanced_internal(root, 0)
        return b

    def is_balanced_internal(self, root, height):
        if not root:
            return height, True
        left_h, left_b = self.is_balanced_internal(root.left, height + 1)
        right_h, right_b = self.is_balanced_internal(root.right, height + 1)
        if abs(left_h - right_h) <= 1 and left_b and right_b:
            return max(left_h, right_h), True
        else:
            return max(left_h, right_h), False


import collections


class Solution_level_order:
    """
    @param root: A Tree
    @return: Level order a list of lists of integer
    """

    def levelOrder(self, root):
        # write your code here
        return self.level_order_internal(root)

    def level_order_internal(self, root):
        s = []  # stack
        res = collections.defaultdict(list)
        if root:
            s.append((root, 0))

        while len(s) > 0:
            node, cur_height = s[0]
            s = s[1:]
            # if cur_height not in res:
            #     res[cur_height] = [node.val]
            # else:
            res[cur_height].append(node.val)
            if node.left:
                s.append((node.left, cur_height + 1))
            if node.right:
                s.append((node.right, cur_height + 1))

        ret = []
        for h, ele in res.items():
            ret.append(ele)
        return ret


class Solution_first_pos_of_targets:
    """
    @param nums: The integer array.
    @param target: Target to find.
    @return: The first position of target. Position starts from 0.
    """

    def binarySearch(self, nums, target):
        # write your code here
        start, end = 0, len(nums) - 1
        res_ind = -1

        while start <= end:
            mid_ind = (start + end) // 2
            mid_val = nums[mid_ind]
            if mid_val == target:
                res_ind = mid_ind
                break
            elif mid_val < target:
                start = mid_ind + 1
            else:
                end = mid_ind - 1

        while res_ind > 0 and nums[res_ind] == nums[res_ind - 1]:
            res_ind -= 1
        return res_ind

class Solution_reverse_words2:
    """
    @param str: a string
    @return: return a string
    """
    def reverseWords(self, str):
        # write your code here
        strings = str.split(' ')
        strings = strings[::-1]
        return ' '.join(strings)

class Solution_max_number_mountainSequence:
    """
    @param nums: a mountain sequence which increase firstly and then decrease
    @return: then mountain top
    """
    def mountainSequence(self, nums):
        # write your code here
        if len(nums) == 0:
            return 0
        max_val = nums[0]
        for val in nums[1:]:
            max_val = max(max_val, val)
        return max_val


class Solution_lca3:
    """
    @param: root: The root of the binary tree.
    @param: A: A TreeNode
    @param: B: A TreeNode
    @return: Return the LCA of the two nodes.
    """

    def lowestCommonAncestor3(self, root, A, B):
        # write your code here
        node, flagA, flagB = self.lca_internal(root, A, B)
        if flagA and flagB:
            return node
        return None

    def lca_internal(self, root, A, B):
        # if not root or root == A or root == B:
        #     return root
        if not root:
            return root, False, False

        # if root == A:
        l, flagAl, flagBl = self.lca_internal(root.left, A, B)
        r, flagAr, flagBr = self.lca_internal(root.right, A, B)

        flagA = root == A
        flagB = root == B

        if flagAl and flagBl:
            return l, True, True
        if flagAr and flagBr:
            return r, True, True
        return root, flagA or flagAl or flagAr, flagB or flagBl or flagBr


class Solution_spiral_index:
    """
    @param matrix: a matrix of m x n elements
    @return: an integer list
    """

    def spiralOrder(self, matrix):
        # write your code here
        res = []
        m = len(matrix)
        if m == 0:
            return res
        n = len(matrix[0])
        if n == 0:
            return res

        up, down, left, right = 0, m - 1, 0, n - 1

        while up <= down and left <= right:
            for i in range(left, right + 1):
                res.append(matrix[up][i])
            for i in range(up + 1, down + 1):
                res.append(matrix[i][right])
            if up == down or left == right:
                break
            for i in range(right - 1, left - 1, -1):
                res.append(matrix[down][i])
            for i in range(down - 1, up, -1):
                res.append(matrix[i][left])
            up += 1
            down -= 1
            left += 1
            right -= 1

        return res


class Solution_buy_stock1:
    """
    @param prices: Given an integer array
    @return: Maximum profit
    """

    def maxProfit(self, prices):
        # write your code here
        profit = 0
        if len(prices) == 0:
            return profit
        val = prices[0]

        for p in prices[1:]:
            if p > val:
                profit = max(profit, p - val)
            else:
                val = p
        return profit


class Solution_buy_sell_stock2:
    """
    @param prices: Given an integer array
    @return: Maximum profit
    """

    def maxProfit(self, prices):
        # write your code here
        profit = 0
        if len(prices) == 0:
            return profit
        val = prices[0]

        for p in prices:
            if p > val:
                profit += p - val
            val = p

        return profit

class Solution_sellbuy3:
    """
    @param prices: Given an integer array
    @return: Maximum profit
    """
    def maxProfit(self, prices):
        # write your code here
        profits = [0]
        if len(prices) == 0:
            return 0
        local_ = [0 for i in range(3)]
        global_ = [0 for i in range(3)]
        for i in range(len(prices)-1):
            diff = prices[i+1] - prices[i]
            for j in range(2, 0, -1):
                local_[j] = max(global_[j-1] + max(diff, 0), local_[j]+diff)
                global_[j] = max(local[j], global_[j])
        return global_[2]

'''
这道是买股票的最佳时间系列问题中最难最复杂的一道，前面两道Best Time to Buy and Sell Stock 买卖股票的最佳时间和Best Time to Buy and Sell Stock II 买股票的最佳时间之二的思路都非常的简洁明了，算法也很简单。而这道是要求最多交易两次，找到最大利润，还是需要用动态规划Dynamic Programming来解，而这里我们需要两个递推公式来分别更新两个变量local和global，参见网友Code Ganker的博客，我们其实可以求至少k次交易的最大利润，找到通解后可以设定 k = 2，即为本题的解答。我们定义local[i][j]为在到达第i天时最多可进行j次交易并且最后一次交易在最后一天卖出的最大利润，此为局部最优。然后我们定义global[i][j]为在到达第i天时最多可进行j次交易的最大利润，此为全局最优。它们的递推式为：

local[i][j] = max(global[i - 1][j - 1] + max(diff, 0), local[i - 1][j] + diff)

global[i][j] = max(local[i][j], global[i - 1][j])

其中局部最优值是比较前一天并少交易一次的全局最优加上大于0的差值，和前一天的局部最优加上差值中取较大值，而全局最优比较局部最优和前一天的全局最优。代码如下：
'''


class Solution_heapify:
    """
    @param: A: Given an integer array
    @return: nothing
    """

    def heapify(self, A):
        # write your code here
        i = len(A) // 2 - 1
        while i >= 0:
            self.heapify_internal(A, i)
            i -= 1

    def heapify_internal(self, A, i):
        left = A[i * 2 + 1] if 2 * i + 1 < len(A) else float('inf')
        right = A[i * 2 + 2] if 2 * i + 2 < len(A) else float('inf')

        if left < right and left < A[i]:
            A[2 * i + 1], A[i] = A[i], A[2 * i + 1]
            self.heapify_internal(A, 2 * i + 1)
            return
        if right < left and right < A[i]:
            A[2 * i + 2], A[i] = A[i], A[2 * i + 2]
            self.heapify_internal(A, 2 * i + 2)

'''topo sort
拓扑排序： 每次将入度为0的节点保存，再从图中删掉，更新剩余节点入度，依次迭代，就可以了。

public ArrayList<DirectedGraphNode> topSort(ArrayList<DirectedGraphNode> graph) {
        // write your code here
        ArrayList<DirectedGraphNode> result = new ArrayList<>();
        HashMap<DirectedGraphNode, Integer> map = new HashMap<>();
        for (DirectedGraphNode node:graph){
            for(DirectedGraphNode ner:node.neighbors){
                if(map.containsKey(ner)){
                    map.put(ner,map.get(ner)+1);
                }else {
                    map.put(ner,1);
                }
            }
        }
        Queue<DirectedGraphNode> q = new LinkedList<>();
        for(DirectedGraphNode node : graph){
            if(!map.containsKey(node)){
                q.offer(node);
                result.add(node);
            }
        }
        while (!q.isEmpty() ){
            DirectedGraphNode node = q.poll();
            for (DirectedGraphNode ner:node.neighbors){
                if(map.get(ner)==1){
                    q.offer(ner);
                    result.add(ner);
                    map.remove(ner);
                }else {
                    map.put(ner, map.get(ner)-1);
                }
            }
        }
        return result;

--------------------- 
作者：cosmos_lee 
来源：CSDN 
原文：https://blog.csdn.net/u012156116/article/details/80981184 
版权声明：本文为博主原创
'''
class Solution_toposort:
    """
    @param: graph: A list of Directed graph node
    @return: Any topological order for the given graph.
    """

    def topSort(self, graph):
        # write your code here
        result = []
        map_ = {}  # map to keep track of the node in degree

        for node in graph:
            for ner in node.neighbors:
                if ner in map_:
                    map_[ner] += 1
                else:
                    map_[ner] = 1

        q = []  # queue
        for node in graph:
            if not node in map_:
                q.append(node)
                result.append(node)

        while len(q) > 0:
            node = q[0]
            q = q[1:]
            for ner in node.neighbors:
                if map_[ner] == 1:
                    q.append(ner)
                    result.append(ner)
                    del map_[ner]
                else:
                    map_[ner] = map_[ner] - 1

        return result


"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""


class Solution_linklist_cycle:
    """
    @param head: The first node of linked list.
    @return: True if it has a cycle, or false
    """

    def hasCycle(self, head):
        # write your code here
        pt1 = pt2 = head

        while pt2 and pt2.next and pt1:
            pt2 = pt2.next.next
            pt1 = pt1.next
            if pt1 == pt2:
                return True
        return False


class Solution_longest_increaseing_subsequence:
    """
    @param nums: An integer array
    @return: The length of LIS (longest increasing subsequence)
    """

    def longestIncreasingSubsequence(self, nums):
        # write your code here
        n = len(nums)
        if n == 0:
            return n
        dp = [1 for i in range(n)]
        max_ = 1

        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
                    max_ = max(dp[i], max_)
        return max_


class Solution_search_in_rotated_sorted_arr:
    """
    @param A: an integer rotated sorted array
    @param target: an integer to be searched
    @return: an integer
    """

    def search(self, A, target):
        # write your code here
        left, right = 0, len(A) - 1
        inds = [(left, right)]

        while len(inds) > 0:
            left, right = inds.pop()
            if left > right:
                continue
            mid = (left + right) // 2
            # print(left, right, mid)
            # mid_val = A[mid_ind]
            if A[mid] == target:
                return mid
            if A[right] == target:
                return right
            if A[left] == target:
                return left

            if A[right] > A[left]:
                if A[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
                inds.append((left, right))
                continue
            if A[right] < A[left]:
                inds.append((left, mid - 1))
                inds.append((mid + 1, right))
        return -1


class Solution_buy_sell_4:
    """
    @param K: An integer
    @param prices: An integer array
    @return: Maximum profit
    """

    def maxProfit(self, K, prices):
        # write your code here
        n = len(prices)
        if n == 0:
            return n
        if K >= n // 2:
            return self.max_profit(prices)
        local_ = [[0 for i in range(K + 1)] for ii in range(n)]
        global_ = [[0 for i in range(K + 1)] for ii in range(n)]

        for i in range(1, len(prices)):
            diff = prices[i] - prices[i - 1]

            for j in range(K, 0, -1):
                local_[i][j] = max(global_[i - 1][j - 1] + max(diff, 0), local_[i - 1][j] + diff)
                global_[i][j] = max(local_[i][j], global_[i - 1][j])

        return global_[-1][-1]

    def max_profit(self, prices):
        res = 0
        for i in range(1, len(prices)):
            res += max(0, prices[i] - prices[i - 1])
        return res

'''
这里我们先解释最多可以进行k次交易的算法，然后最多进行两次我们只需要把k取成2即可。我们还是使用“局部最优和全局最优解法”。我们维护两种量，
一个是当前到达第i天可以最多进行j次交易，最好的利润是多少（global[i][j]），另一个是当前到达第i天，最多可进行j次交易，
并且最后一次交易在当天卖出的最好的利润是多少（local[i][j]）。下面我们来看递推式。

全局的比较简单，
global[i][j]=max(local[i][j],global[i-1][j])，
也就是去当前局部最好的，和过往全局最好的中大的那个（因为最后一次交易如果包含当前天一定在局部最好的里面，否则一定在过往全局最优的里面）。

对于局部变量的维护，递推式是
local[i][j]=max(global[i-1][j-1]+max(diff,0),local[i-1][j]+diff)，
也就是看两个量，第一个是全局到i-1天进行j-1次交易，然后加上今天的交易，如果今天是赚钱的话（也就是前面只要j-1次交易，最后一次交易取当前天），
第二个量则是取local第i-1天j次交易，然后加上今天的差值（这里因为local[i-1][j]比如包含第i-1天卖出的交易，所以现在变成第i天卖出，并不会增加交易次数，
而且这里无论diff是不是大于0都一定要加上，因为否则就不满足local[i][j]必须在最后一天卖出的条件了）。


上面的算法中对于天数需要一次扫描，而每次要对交易次数进行递推式求解，所以时间复杂度是O(n*k)，如果是最多进行两次交易，那么复杂度还是O(n)。
空间上只需要维护当天数据皆可以，所以是O(k)，当k=2，则是O(1)。"
--------------------- 
作者：feliciafay 
来源：CSDN 
原文：https://blog.csdn.net/feliciafay/article/details/45128771 
版权声明：本文为博主原创文章，转载请附上博文链接！
'''


class DListNode:
    def __init__(self, k=-1, v=0, prev_=None, next_=None):
        self.k = k
        self.v = v
        self.prev = prev_
        self.next = next_


class hashed_linked_list:
    def __init__(self, capacity):
        self.head = DListNode()
        self.tail = DListNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.length = 0
        self.capacity = capacity
        self.dic = {}

    def __len__(self):
        return self.length

    def add_front(self, k, v):
        '''
            add them to the front and evict the last if len > capacity
        '''
        node = DListNode(k, v)
        # insert the node into head
        head_next = self.head.next
        node.prev = self.head
        node.next = head_next
        head_next.prev = node
        self.head.next = node
        self.dic[k] = node
        self.length += 1
        if self.length > self.capacity:
            '''evict the last one'''
            self.pop_end()

    def pop_end(self):
        if self.length == 0:
            raise Exception
        tail_prev = self.tail.prev
        tail_prev_prev = tail_prev.prev
        tail_prev_prev.next = self.tail
        self.tail.prev = tail_prev_prev
        del self.dic[tail_prev.k]
        del tail_prev
        self.length -= 1

    def move_front(self, k, value=None):
        assert k in self.dic
        node = self.dic[k]
        node_prev = node.prev
        node_next = node.next
        node_prev.next = node_next
        node_next.prev = node_prev
        node.next = None

        # move the node to the front
        head_next = self.head.next
        node.prev = self.head
        node.next = head_next
        head_next.prev = node
        self.head.next = node

        if value:
            node.v = value

    def search(self, k):
        if k in self.dic:
            return self.dic[k]
        else:
            return None

    def expr(self):
        ret = []
        pt = self.head.next
        while pt != self.tail:
            ret.append((pt.k, pt.v))
            pt = pt.next
        return str(ret), str(self.dic)

    def __unicode__(self):
        return self.expr()

    def __repr__(self):
        return str(self.expr())


class LRUCache_lintcode:
    """
    @param: capacity: An integer
    """

    # linked -hash table solution
    def __init__(self, capacity):
        # do intialization if necessary
        self.l = hashed_linked_list(capacity)

    """
    @param: key: An integer
    @return: An integer
    """

    def get(self, key):
        # write your code here
        if self.l.search(key):
            ret = self.l.search(key).v
            self.l.move_front(key)
            return ret
        else:
            return -1

    """
    @param: key: An integer
    @param: value: An integer
    @return: nothing
    """

    def set(self, key, value):
        if self.l.search(key):
            self.l.move_front(key, value)
        else:
            self.l.add_front(key, value)

class Solution_eggdrop:
    """
    @param n: An integer
    @return: The sum of a and b
    """
    def dropEggs(self, n):
        # write your code here
        i = 0
        acc = 0
        while acc < n:
            i += 1
            acc += i

        return i

'''
因为只有两个鸡蛋，所以第一个鸡蛋应该是按一定的间距扔，比如10楼，20楼，30楼等等，比如10楼和20楼没碎，30楼碎了，那么第二个鸡蛋就要做线性搜索，
分别尝试21楼，22楼，23楼等等直到鸡蛋碎了，就能找到临界点。那么我们来看下列两种情况：

1. 假如临界点是9楼，那么鸡蛋1在第一次扔10楼碎掉，然后鸡蛋2依次遍历1到9楼，则总共需要扔10次。

2. 假如临界点是100楼，那么鸡蛋1需要扔10次，到100楼时碎掉，然后鸡蛋2依次遍历91楼到100楼，总共需要扔19次。

所以上述方法的最坏情况是19次，那么有没有更少的方法呢，上面那个方法每多扔一次鸡蛋1，鸡蛋2的线性搜索次数最多还是10次，
那么最坏情况肯定会增加，所以我们需要让每多扔一次鸡蛋1，鸡蛋2的线性搜索最坏情况减少1，这样恩能够保持整体最坏情况的平衡，那么我们假设鸡蛋1第一次在第X层扔，然后向上X-1层扔一次，然后向上X-2层扔，以此类推直到100层，那么我们通过下面的公式求出X：

X + (X-1) + (X-2) + ... + 1 = 100 -> X = 14

所以我们先到14楼，然后27楼，然后39楼，以此类推，最坏情况需要扔14次。
'''


class Solution_getABsubstring:
    """
    @param S: a String consists of a and b
    @return: the longest of the longest string that meets the condition
    """

    def getAns(self, S):
        # Write your code here
        dic = {}
        count = 0
        dic[count] = -1
        res = 0

        for ind, s in enumerate(S):
            count = count + 1 if s == 'A' else count - 1
            if not count in dic:
                dic[count] = ind
            else:
                res = max(res, ind - dic[count])

        return res


import copy

import copy


class Solution_combination_sum:
    """
    @param candidates: A list of integers
    @param target: An integer
    @return: A list of lists of integers
    """

    def combinationSum(self, candidates, target):
        # write your code here
        res = []
        candidates.sort()
        candidates = self.remove_dep(candidates)
        self.combination_sum_internal(candidates, 0, [], target, res)
        return res

    def combination_sum_internal(self, candidates, ind, cur_res, target, res):
        if target == 0:
            res.append(cur_res)
            return
        if target < 0:
            return

        i = ind
        while i < len(candidates):
            cur_res.append(candidates[i])
            self.combination_sum_internal(candidates, i, copy.deepcopy(cur_res),
                                          target - candidates[i], res)
            cur_res.pop()
            i += 1

    def remove_dep(self, candidates):
        ind = 1
        for i in range(1, len(candidates)):
            if candidates[i] != candidates[i - 1]:
                candidates[ind] = candidates[i]
                ind += 1
        return candidates[:ind]


class Solution_comb_sum2:
    """
    @param num: Given the candidate numbers
    @param target: Given the target number
    @return: All the combinations that sum to target
    """

    def combinationSum2(self, num, target):
        # write your code here
        res = []
        num.sort()
        self.combination_sum_internal(num, 0, [], res, target)
        return res

    def combination_sum_internal(self, num, ind, cur_sol, res, target):
        if target == 0:
            res.append(cur_sol)
            return
        if target < 0:
            return

        i = ind
        while i < len(num):
            cur_sol.append(num[i])
            self.combination_sum_internal(num, i + 1, copy.deepcopy(cur_sol), res,
                                          target - num[i])
            cur_sol.pop()
            while i < len(num) - 1 and num[i] == num[i + 1]:
                i += 1
            i += 1


class Solution_hanoi:
    """
    @param n: the number of disks
    @return: the order of moves
    """

    def towerOfHanoi(self, n):
        # write your code here
        res = []
        self.hanoi_internal(n, 'A', 'C', 'B', res)
        return res

    def hanoi_internal(self, n, from_, to_, other_, res):
        if n == 1:
            res.append("from {} to {}".format(from_, to_))
            return
        self.hanoi_internal(n - 1, from_, other_, to_, res)
        res.append("from {} to {}".format(from_, to_))
        self.hanoi_internal(n - 1, other_, to_, from_, res)


import copy


class Solution_letter_combination:
    """
    @param digits: A digital string
    @return: all posible letter combinations
    """

    def init_dic(self):
        self.dic = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }

    def letterCombinations(self, digits):
        # write your code here
        if not hasattr(self, 'dic'):
            self.init_dic()

        res = []
        if len(digits) == 0:
            return res
        self.letter_combinations_internal(digits, 0, '', res)
        return res

    def letter_combinations_internal(self, digits, ind, cur_sol, res):
        if ind >= len(digits):
            res.append(cur_sol)
            return

        for char in self.dic[digits[ind]]:
            self.letter_combinations_internal(digits, ind + 1, cur_sol + char, res)


import copy


class Solution_gen_parenthesis:
    """
    @param n: n pairs
    @return: All combinations of well-formed parentheses
    """

    def generateParenthesis(self, n):
        # write your code here
        res = []

        self.gen_internal('(', n, 1, 0, res)
        return res

    def gen_internal(self, cur, n, left, right, res):
        if right > left:
            return
        if len(cur) == 2 * n and left == right:
            res.append(cur)
        if left < n:
            self.gen_internal(cur + "(", n, left + 1, right, res)
        if right < n:
            self.gen_internal(cur + ")", n, left, right + 1, res)


class Solution_restore_ip_addr:
    """
    @param s: the IP string
    @return: All possible valid IP addresses
    """

    def restoreIpAddresses(self, s):
        # write your code here
        if len(s) <= 3:
            return []
        if s == "0000":
            return ["0.0.0.0"]
        res = []
        self.restore_internal(s, [], res)
        return res

    def is_block(self, s):
        if len(s) > 3 or len(s) <= 0:
            return False

        if s[0] == "0" and len(s) > 1:
            return False
        elif int(s) >= 0 and int(s) <= 255:
            return True
        else:
            return False

    def restore_internal(self, s, cur_res, res):
        if self.is_block(s) and len(cur_res) == 3:
            cur_res.append(s)
            res.append(".".join(cur_res))

        i = 1
        while i <= 3 and i < len(s):
            block_, next_s = s[:i], s[i:]
            if self.is_block(block_):
                self.restore_internal(next_s, cur_res + [block_], res)
            i += 1


class Solution_spltstr:
    """
    @param: : a string to be split
    @return: all possible split string array
    """

    def splitString(self, s):
        # write your code here
        res = []
        self.split_str_internal(s, 0, [], res)
        return res

    def split_str_internal(self, s, ind, cur_sol, res):
        if ind == len(s):
            res.append(cur_sol)
            return

        if ind > len(s):
            return

        self.split_str_internal(s, ind + 1, cur_sol + [s[ind: ind + 1]], res)
        self.split_str_internal(s, ind + 2, cur_sol + [s[ind: ind + 2]], res)


import copy


class Solution_min_stickers:
    """
    @param stickers: a string array
    @param target: a string
    @return: the minimum number of stickers that you need to spell out the target
    """

    def minStickers(self, stickers, target):
        # Write your code here
        self.res = float('inf')
        stickers = list(map(lambda x: self.word2dic(x), stickers))
        target = self.word2dic(target)
        self.min_stickers_internal_no_ind(stickers, target, 0)
        if self.res == float('inf'):
            self.res = -1
        return self.res

    def word2dic(self, word):
        ret = collections.defaultdict(int)
        for w in word:
            ret[w] += 1
        return ret

    def eleminate(self, s, target):
        ''' remove any chars seen by s and return the left over target'''
        can_eleminate = False

        for c in s:
            for ind_t, c_t in enumerate(target):
                if c == c_t:
                    can_eleminate = True
                    target = target[:ind_t] + target[ind_t + 1:]
                    break
        return can_eleminate, target

    def eleminate_dic(self, s, target):
        ''' remove any chars seen by s and return the left over target'''
        can_eleminate = False
        for c in s.keys():
            if c in target and target[c] > 0:
                can_eleminate = True
                target[c] -= s[c]
                if target[c] <= 0:
                    del target[c]
        return can_eleminate, target

    def min_stickers_internal(self, stickers, ind, target, cur_res):
        if len(target) == 0:
            self.res = min(self.res, len(cur_res))

        i = ind
        while i < len(stickers):
            can_eleminate, new_target = self.eleminate_dic(stickers[i], copy.deepcopy(target))
            if can_eleminate:
                self.min_stickers_internal(stickers, ind, new_target, cur_res + [stickers[i]])
            i += 1

    def min_stickers_internal_no_ind(self, stickers, target, cur_res):
        if len(target) == 0:
            self.res = min(self.res, cur_res)

        i = 0
        while i < len(stickers):
            can_eleminate, new_target = self.eleminate_dic(stickers[i], copy.deepcopy(target))
            if can_eleminate:
                self.min_stickers_internal_no_ind(stickers[i:], new_target, cur_res + 1)
            i += 1


class Solution_combinations:
    """
    @param n: Given the range of numbers
    @param k: Given the numbers of combinations
    @return: All the combinations of k numbers out of 1..n
    """

    def combine(self, n, k):
        # write your code here
        res = []
        self.combine_internal([i + 1 for i in range(n)], k, [], res)
        return res

    def combine_internal(self, n_list, k, cur_res, res):
        if len(cur_res) == k:
            res.append(cur_res)

        for ind, num in enumerate(n_list):
            self.combine_internal(n_list[ind + 1:], k, cur_res + [num], res)

class Solution_build_segment_tree:
    """
    @param: start: start value.
    @param: end: end value.
    @return: The root of Segment Tree.
    """
    def build(self, start, end):
        # write your code here
        if start > end:
            return None
        root = SegmentTreeNode(start, end)
        if start != end:
            root.left = self.build(start, (start+end)//2)
            root.right = self.build((start+end)//2+1, end)
        return root


class _SegmentTreeNode:
    def __init__(self, start, end, value=None):
        self.start, self.end, self.value = start, end, value
        self.left, self.right = None, None


class Solution_query_interval:
    """
    @param A: An integer array
    @param queries: An query list
    @return: The result list
    """

    def build(self, start, end, A):
        # write your code here
        if start > end:
            return None
        root = _SegmentTreeNode(start, end)
        if start != end:
            root.left = self.build(start, (start + end) // 2, A)
            root.right = self.build((start + end) // 2 + 1, end, A)
            root.min = min(root.left.min, root.right.min)
        else:
            root.min = A[start]
        return root

    def query(self, root, start, end):
        if start == root.start and end == root.end:
            return root.min
        mid = (root.start + root.end) // 2
        assert start <= end, "input error"
        if end <= mid:
            return self.query(root.left, start, end)
        elif start >= mid + 1:
            return self.query(root.right, start, end)
        else:
            return min(self.query(root.left, start, mid),
                       self.query(root.right, mid + 1, end))

    def intervalMinNumber(self, A, queries):
        # write your code here
        root = self.build(0, len(A) - 1, A)
        res = []
        for q in queries:
            res.append(self.query(root, q.start, q.end))
        return res


class Solution_query_seg_tree:
    """
    @param root: The root of segment tree.
    @param start: start value.
    @param end: end value.
    @return: The maximum number in the interval [start, end]
    """

    def query(self, root, start, end):
        # write your code here
        mid = (root.start + root.end) // 2
        if start > end:
            return -1

        if start == root.start and end == root.end:
            return root.max
        if end <= mid:
            return self.query(root.left, start, end)
        elif start >= mid + 1:
            return self.query(root.right, start, end)
        else:
            return max(self.query(root.left, start, mid),
                       self.query(root.right, mid + 1, end))


class Solution_modify_seg_tree:
    """
    @param root: The root of segment tree.
    @param index: index.
    @param value: value
    @return: nothing
    """

    def modify(self, root, index, value):
        # write your code here
        if index == root.start and index == root.end:
            root.max = value
        if root.start == root.end:
            return
        mid = (root.start + root.end) // 2
        if index < root.start or index > root.end:
            return
        elif index <= mid:
            self.modify(root.left, index, value)
        else:
            self.modify(root.right, index, value)
        max_left = root.left.max
        max_right = root.right.max
        root.max = max(max_left, max_right)


class Solution_sum_interval_tree:
    def build(self, start, end, A):
        # write your code here
        if start > end:
            return None
        root = _SegmentTreeNode(start, end)
        if start != end:
            root.left = self.build(start, (start + end) // 2, A)
            root.right = self.build((start + end) // 2 + 1, end, A)
            root.sum = root.left.sum + root.right.sum
        else:
            root.sum = A[start]
        return root

    def query(self, root, start, end):
        if start == root.start and end == root.end:
            return root.sum
        mid = (root.start + root.end) // 2
        assert start <= end, "input error"
        if end <= mid:
            return self.query(root.left, start, end)
        elif start >= mid + 1:
            return self.query(root.right, start, end)
        else:
            return self.query(root.left, start, mid) + \
                   self.query(root.right, mid + 1, end)

    """
    @param A: An integer list
    @param queries: An query list
    @return: The result list
    """

    def intervalSum(self, A, queries):
        root = self.build(0, len(A) - 1, A)
        res = []
        for q in queries:
            res.append(self.query(root, q.start, q.end))
        return res


class Solution_lcs:
    """
    @param A: A string
    @param B: A string
    @return: The length of longest common subsequence of A and B
    """

    def longestCommonSubsequence(self, A, B):
        # write your code here
        la, lb = len(A), len(B)
        dp = [[0 for i in range(lb + 1)] for j in range(la + 1)]

        for i in range(1, la + 1):
            for j in range(1, lb + 1):
                if A[i - 1] == B[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])

        return dp[-1][-1]


class Solution_interleave:
    """
    @param s1: A string
    @param s2: A string
    @param s3: A string
    @return: Determine whether s3 is formed by interleaving of s1 and s2
    """

    def isInterleave(self, s1, s2, s3):
        # write your code here
        l1 = len(s1)
        l2 = len(s2)
        if l1 + l2 != len(s3):
            return False
        dp = [[False for i in range(l2 + 1)] for j in range(l1 + 1)]
        dp[0][0] = True
        for i in range(l1):
            dp[i + 1][0] = (dp[i][0] and s1[i] == s3[i])

        for j in range(l2):
            dp[0][j + 1] = (dp[0][j] and s2[j] == s3[j])

        for i in range(l1):
            for j in range(l2):
                dp[i + 1][j + 1] = (dp[i + 1][j] and s2[j] == s3[i + j + 1]) or \
                                   (dp[i][j + 1] and s1[i] == s3[i + j + 1])
        return dp[l1][l2]

'''cpp
class Solution {
public:
    /**
     * @param m: An integer m denotes the size of a backpack
     * @param A: Given n items with size A[i]
     * @return: The maximum size
     */
    int backPack(int m, vector<int> &A) {
        // write your code here
        int n = A.size();
        int dp[m+1] = {0};
        
        for (int i = 0; i < n; ++i) {
            for (int j = m - 1; j >= 0; --j) {
                if (j + 1 - A[i] >= 0) {
                    dp[j+1] = max(dp[j+1-A[i]] + A[i], dp[j+1]);
                }
            }
        }
        
        return dp[m];
    }
};
'''


class Solution_coin_change:
    """
    @param coins: a list of integer
    @param amount: a total amount of money amount
    @return: the fewest number of coins that you need to make up
    """

    def coinChange(self, coins, amount):
        # write your code here
        n = len(coins)
        m = amount

        # dp = [[0 for i in range(m+1)] for j in range(m+1)]
        dp = [m + 1 for i in range(m + 1)]
        dp[0] = 0

        for i in range(amount + 1):
            for j in range(n):
                if coins[j] <= i:
                    dp[i] = min(dp[i], dp[i - coins[j]] + 1)

        return -1 if dp[m] > m else dp[m]

class Solution_cut_rope:
    """
    @param prices: the prices
    @param n: the length of rod
    @return: the max value
    """

    def cutting(self, prices, n):
        # Write your code here
        dp = [0 for i in range(n + 1)]
        dp[0] = 0

        for i in range(1, n + 1):
            for j in range(i):
                dp[i] = max(dp[i], dp[j] + prices[i - 1 - j])

        return dp[n]


class Solution_johns_backyard_kindergarden:
    """
    @param x: the wall's height
    @return: YES or NO
    """

    def isBuild(self, x):
        # write you code here
        dp = [False for i in range(x + 1)]
        if x < 3 or (x > 3 and x < 7):
            return "NO"
        if x == 3 or x == 6:
            return "YES"
        dp[3] = True
        dp[6] = True
        dp[7] = True

        for i in range(8, x + 1):
            dp[i] = dp[i - 3] or dp[i - 7]

        return "YES" if dp[x] else "NO"


class NumArray_(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.arr = [0 for i in range(len(nums))]
        self.bit = [0 for i in range(len(nums) + 1)]

        for ind, num in enumerate(nums):
            self.update(ind, num)

    def update(self, index, val):
        """
        :type i: int
        :type val: int
        :rtype: void
        """
        delta = val - self.arr[index]
        self.arr[index] = val

        i = index + 1
        # for i in range(index, len(self.arr)+1, self.lowbit(i)):
        while i <= len(self.arr):
            self.bit[i] += delta
            i += self.lowbit(i)

    def lowbit(self, x):
        return x & (-x)

    def getPrefixSum(self, index):
        sum_ = 0
        i = index + 1
        # for i in range(index+1, -1, -self.lowbit(i)):
        while i > 0:
            sum_ += self.bit[i]
            i -= self.lowbit(i)
        return sum_

    def sumRange(self, left, right):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.getPrefixSum(right) - self.getPrefixSum(left - 1)

'''
我们都知道， -x 的值， 其实就是在x的值的基础上进行按位取反(~x)之后在增加1所得， 也就是说， 
x & -x == x & (~x + 1)

x & -x 表示最低位的为1 表示的二进制数
https://www.geeksforgeeks.org/binary-indexed-tree-or-fenwick-tree-2/
'''

'''
位运算实现 加减乘除 https://blog.csdn.net/ojshilu/article/details/11179911
'''

'''
minimum spanning tree
* prim algorithm
在所有u∈Vt，v∈V-Vt的边(u,v)∈E中找出一条代价最小的边(u0,v0)并入集合Et，通知将v0并入Vt，直至Vt=V为止。
此时Et中必有n-1条边，则T=(Vt,Et)为N的最小生成树。
* kruskal algorithm
初始化：Vt=V，Et=∅。即每个顶点构成一棵独立的树，T此时是一个仅含|V|个顶点的森林；
循环（重复下列操作直至T是一棵树）：按G的边的权值递增顺序依次从E-Et中选择一条边，
如果这条边加入T后不构成回路，则将其加入Et，否则舍弃，直至Et中含有n-1条边。
'''

from functools import cmp_to_key


class UnionFind:
    def __init__(self, n):
        self.father = [i for i in range(n)]

    def find(self, x):
        if self.father[x] == x:
            return x
        self.father[x] = self.find(self.father[x])
        return self.father[x]

    def connect(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.father[root_a] = root_b


class Solution_mst_prim:
    # @param {Connection[]} connections given a list of connections
    # include two cities and cost
    # @return {Connection[]} a list of connections from results
    def connection_cmp(self, connection1, connection2):
        if connection1.cost != connection2.cost:
            return connection1.cost - connection2.cost

        if connection1.city1 != connection2.city1:
            return 1 if connection1.city1 >= connection2.city1 else -1

        return 1 if connection1.city2 >= connection2.city2 else -1

    def lowestCost(self, connections):
        # Write your code here
        mst = []
        if connections == None or len(connections) == 0:
            return mst
        connections.sort(key=cmp_to_key(self.connection_cmp))
        idx = 0
        strToIdxMap = {}
        for c in connections:
            if c.city1 not in strToIdxMap:
                strToIdxMap[c.city1] = idx
                idx += 1
            if c.city2 not in strToIdxMap:
                strToIdxMap[c.city2] = idx
                idx += 1
        uf = UnionFind(idx)

        for c in connections:
            city1Root = uf.find(strToIdxMap[c.city1])
            city2Root = uf.find(strToIdxMap[c.city2])
            if city1Root != city2Root:
                mst.append(c)
                uf.connect(city1Root, city2Root)

        if (len(mst) < idx - 1):
            return []
        else:
            return mst
# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(i,val)
# param_2 = obj.sumRange(i,j)
class Solution_search_2dmatrix1:
    """
    @param matrix: matrix, a list of lists of integers
    @param target: An integer
    @return: a boolean, indicate whether matrix contains target
    """

    def searchMatrix(self, matrix, target):
        # write your code here
        row = len(matrix)
        if row == 0:
            return False
        col = len(matrix[0])
        if col == 0:
            return False

        left, right = 0, row - 1
        ind = -1
        while left <= right:
            mid = (left + right) // 2

            if matrix[mid][0] < target:
                left = mid + 1
            elif matrix[mid][0] > target:
                right = mid - 1
            else:
                return True

        if right == -1:
            return False
        ind = right
        left, right = 0, col - 1

        while left <= right:
            mid = (left + right) // 2

            if matrix[ind][mid] < target:
                left = mid + 1
            elif matrix[ind][mid] > target:
                right = mid - 1
            else:
                return True

        return False


class Solution:
    """
    @param matrix: A list of lists of integers
    @param target: An integer you want to search in matrix
    @return: An integer indicate the total occurrence of target in the given matrix
    """

    def bin_search(self, row, target):
        left, right = 0, len(row) - 1
        ind = -1
        while left <= right:
            mid = (left + right) // 2

            if row[mid] < target:
                left = mid + 1
            elif row[mid] > target:
                right = mid - 1
            else:
                return True
        return False

    def searchMatrix(self, matrix, target):
        # write your code here
        res = 0
        for row in matrix:
            if self.bin_search(row, target):
                res += 1
        return res


class Solution_485:
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        cur_res = 0
        for num in nums:
            if num == 1:
                cur_res += 1
                res = max(res, cur_res)
            else:
                cur_res = 0
        return res

'''
Kadane's algorithm
'''
class Solution_max_subarray:
    """
    @param nums: A list of integers
    @return: A integer indicate the sum of max subarray
    """
    def maxSubArray(self, nums):
        # write your code here
        res, cur_sum = float('-inf'), 0
        for num in nums:
            cur_sum = max(cur_sum + num, num)
            res = max(res, cur_sum)
        return res

'''
public static void swap(int a, int b) {
    a = b - a; // 9 - 5 = 4
    b = b - a; // 9 - 4 = 5
    a = a + b; // 4 + 5 = 9
}
    System.out.println(“a: “ + a);
    System.out.println(“b: “ + b);
'''
def swap(a, b):
    a = b - a
    b = b - a
    a = a + b
    return a, b

class Solution_minimum_subarray:
    """
    @param: nums: a list of integers
    @return: A integer indicate the sum of minimum subarray
    """
    def minSubArray(self, nums):
        # write your code here
        min_ = 0
        res = float('inf')

        for i in range(0, len(nums)):
            min_ = min(min_ + nums[i], nums[i])
            res = min(res, min_)

        return res

class Solution_inorder_traverse:
    """

    @param root: A Tree
    @return: Inorder in ArrayList which contains node values.
    """
    def inorderTraversal(self, root):
        # write your code here
        # res = []
        # self.inorder_recursive_internal(root, res)
        # return res
        q = []
        res = []
        if root:
            q.append((root, False))
        
        while len(q) > 0:
            node, visited = q.pop()
            # print(node.val, visited)
            if visited:
                res.append(node.val)
                continue
            
            if node.right:
                q.append((node.right, False))
            q.append((node, True))
            if node.left:
                q.append((node.left, False))
        return res
            
                
        
    def inorder_recursive_internal(self, root, res):
        if not root:
            return
        
        self.inorder_recursive_internal(root.left, res)
        res.append(root.val)
        self.inorder_recursive_internal(root.right, res)

if __name__ == "__main__":
    # lru = LRUCache_lintcode(2)
    # lru.set(2, 1)
    # lru.set(1, 1)
    # print(lru.get(2))
    # lru.set(4, 1)
    # print(lru.get(1))
    # print(lru.get(2))
    # sol = Solution_comb_sum()
    # sol.combinationSum([2, 2, 3, 7], 7)
    sol = Solution_min_stickers()
    stickers = ["when","hard","spot","window","know","above","cloud","go","loud","bank"]
    stickers1 = ["swim","cry","sight","less","dead","except","jump","course","pound","us","laugh","need","milk","subject","thank","no","pay","tube","sail","snow","dont","way","party","was","friend","poor","picture","spell","parent","separate","joy","safe","wave","post","stone","room","whether","straight","clock","dog","bed","element","large","this","shore","street","truck","nose","team","my"]
    target = "caselight"
    target1 = 'asact'
    ret = sol.minStickers(stickers1, target1)
    print(ret)
