class Solution_785:
    def isBipartite(self, graph):
        """
        leetcode 785 accepted
        :type graph: List[List[int]]
        :rtype: bool
        """
        vert_id = 0
        color = [0 for i in range(len(graph))]
        for vert_id, neighbor in enumerate(graph):
            if color[vert_id] == 0:
                if not self.dfs(graph, vert_id, color, 1):
                    return False
        return True
                
    def dfs(self, graph, vert_id, color, color_val):
        color[vert_id] = color_val
        for n in graph[vert_id]:          
            if color[n] == color_val:
                return False
            if color[n] == 0 and \
                not self.dfs(graph, n, color, -color_val):
                return False
        return True    

import copy

class Solution_78(object):
    def subsets(self, nums):
        """
        leetcode 78 subsets
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        self.subsets_internal(nums, 0, [], res)
        return res
        
    def subsets_internal(self, nums, ind, cur_res, res):
        if ind == len(nums):
            res.append(cur_res)
            return 
        
        cur_res.append(nums[ind])
        self.subsets_internal(nums, ind+1, copy.deepcopy(cur_res), res)    
        cur_res = cur_res[:-1]
        self.subsets_internal(nums, ind+1, copy.deepcopy(cur_res), res)

class Solution_78_iterative(object):
    def subsets(self, nums):
        """
        leetcode 78 subsets
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = [[]]
        nums = sorted(nums)

        for i in nums:
            res_size = len(result)
            for one_res in result[:res_size]:
                one_res = copy.deepcopy(one_res)
                one_res.append(i)
                result.append(one_res)

        return result

class Solution_46(object):
    
    def permute(self, nums):
        """
        leetcode 46 permutations
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) == 0 and nums is None:
            return result
        result = []
        self.permute_internal(nums, 0, len(nums), [], result)
        return result
        
    def permute_internal(self, nums, ind, length, cur_res, result):
        if ind == length:
            result.append(cur_res)
            return
         
        for i in nums:
            one_res = copy.deepcopy(cur_res)   
            new_nums = copy.deepcopy(nums)
            one_res.append(i)
            new_nums.remove(i)
            self.permute_internal(new_nums, ind+1, length, one_res, result)


