class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root:
            current = root
            stack = []
            final_list = list()

            while True:
                if current is not None:
                    stack.append(current)
                    current = current.left
                elif stack:
                    current = stack.pop()
                    final_list.append(current.val)
                    current = current.right
                    
                else:
                    break
            return final_list
