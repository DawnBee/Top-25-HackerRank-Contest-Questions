######################
# MAXIMUM PASSENGERS #
######################

# Matrix
# Path Finding
# Depth-First Search

# ChatGPT's Approach
'''
from collections import deque

def max_passenger(matrix):
    n = len(matrix)
    
    if matrix[0][0] == -1 or matrix[n-1][n-1] == -1:
        return 0  # Start or end is blocked, no passengers can be picked

    def dfs(x, y, can_pick):
        if x < 0 or y < 0 or x >= n or y >= n or matrix[x][y] == -1:
            return 0  # Out of bounds or blocked cell
        
        passengers = matrix[x][y]
        matrix[x][y] = -1  # Mark the cell as visited
        
        # Explore right and down directions
        right = dfs(x, y + 1, can_pick)
        down = dfs(x + 1, y, can_pick)
        
        # Reset the cell after the right and down directions
        matrix[x][y] = passengers
        
        if can_pick:
            result = passengers + max(right, down)
        else:
            result = max(right, down)
        
        return result
    
    forward_trip = dfs(0, 0, True)
    backward_trip = dfs(n - 1, n - 1, False)
    
    return forward_trip + backward_trip
'''


# My Approach
'''
def max_passenger(matrix):
    rows, cols = len(matrix), len(matrix[0])
    result = 0

    # To Railway Station (Right & Down)
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                matrix[i][j] = 0
                result += 1
            elif matrix[i][j] == -1:
                break

    # To Starting Point (Left & Top)
    for i in range(rows-1,-1,-1):
        for j in range(cols-1,-1,-1):
            if matrix[i][j] == 1:
                matrix[i][j] = 0
                result += 1
            elif matrix[i][j] == -1:
                break

    return result

mat_1 = [
    [0, 1],
    [-1, 0]
]

mat_2 = [
    [0,0,0,1],
    [1,-1,0,0],
    [1,0,0,0],
    [0,0,-1,1],
]

mat_3 = [
    [0,0,0,1],
    [1,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
]

mat_4 = [
    [1,0,0,-1,0,1],
    [1,0,0,0,1,0],
    [0,0,-1,0,0,1],
    [1,0,0,1,0,0],
    [0,0,0,0,0,0],
    [0,1,-1,0,0,0],        
]

print(max_passenger(mat_4))
'''


# 1 0 0 -1 0 1
# 1 0 0 0 1 0
# 0 0 -1 0 0 1
# 1 0 0 1 0 0
# 0 0 0 0 0 0
# 0 1 -1 0 0 -1
# Result : 0

# 1 0 0 -1 0 1
# 1 0 0 0 1 0
# 0 0 -1 0 0 1
# 1 0 0 1 0 0
# 0 0 0 0 0 0
# 0 1 -1 0 0 0
# Result: 6

# 0 0 0 1
# 1 0 0 0
# 0 0 0 0
# 0 0 0 0
# Result 2

# 0 0 0 1
# 1 -1 0 0
# 1 0 0 0
# 0 0 -1 1
# Result: 4


# From the Internet
'''
def cost(grid, row1, col1, row2, col2):
    if row1 == row2 and col1 == col2:
        if grid[row1][col1] == 1:
            return 1
        return 0

    ans = 0

    if grid[row1][col1] == 1:
        ans += 1

    if grid[row2][col2] == 1:
        ans += 1

    return ans


def solve(n, m, grid, dp, row1, col1, row2):
    col2 = (row1 + col1) - (row2)

    if row1 == n - 1 and col1 == m - 1 and row2 == n - 1 and col2 == m - 1:
        return 0

    if row1 >= n or col1 >= m or row2 >= n or col2 >= m:
        return -float("inf")

    if dp[row1][col1][row2] != -1:
        return dp[row1][col1][row2]

    ch1 = ch2 = ch3 = ch4 = -float("inf")

    if col1 + 1 < m and col2 + 1 < m and row1 + 1 < n and row2 + 1 < n and grid[row1][col1 + 1] != -1 and grid[row2 + 1][col2 + 1] != -1:
        ch1 = cost(grid, row1, col1 + 1, row2 + 1, col2 + 1) + solve(n, m, grid, dp, row1, col1 + 1, row2 + 1)

    if col1 + 1 < m and col2 + 1 < m and row1 + 1 < n and row2 < n and grid[row1][col1 + 1] != -1 and grid[row2][col2] != -1:
        ch2 = cost(grid, row1, col1 + 1, row2, col2) + solve(n, m, grid, dp, row1, col1 + 1, row2)

    if row1 + 1 < n and col2 + 1 < m and row2 < n and grid[row1 + 1][col1] != -1 and grid[row2][col2] != -1:
        ch3 = cost(grid, row1 + 1, col1, row2, col2) + solve(n, m, grid, dp, row1 + 1, col1, row2)

    if row1 + 1 < n and row2 + 1 < n and col1 + 1 < m and col2 < m and grid[row1 + 1][col1] != -1 and grid[row2 + 1][col2] != -1:
        ch4 = cost(grid, row1 + 1, col1, row2 + 1, col2) + solve(n, m, grid, dp, row1 + 1, col1, row2 + 1)

    dp[row1][col1][row2] = max(ch1, max(ch2, max(ch3, ch4)))
    return dp[row1][col1][row2]


def initialize_dp(dp, item):
    n = len(dp)
    m = len(dp[0])
    p = len(dp[0][0])

    for i in range(n):
        for j in range(m):
            for k in range(p):
                dp[i][j][k] = item

def collect_max(n, m, grid):
    ans = 0
    dp = [[[[-1] * 6 for _ in range(6)] for _ in range(6)] for _ in range(6)]
    initialize_dp(dp, -1)

    if grid[n - 1][m - 1] == -1 or grid[0][0] == -1:
        ans = -float("inf")

    if grid[0][0] == 1:
        ans += 1
    grid[0][0] = 0

    if grid[n - 1][m - 1] == 1:
        ans += 1
    grid[n - 1][m - 1] = 0

    ans += solve(n, m, grid, dp, 0, 0, 0)
    return max(ans, 0)


if __name__ == '__main__':
    n = int(input())
    m = int(input())
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().split())))
    print(collect_max(n, m, arr))
'''

#########################
# MINIMUM STREET LIGHTS #
#########################

'''
def min_lights_to_cover_road(n, locations):
    # Create a list of ranges covered by each light
    ranges = []
    for i in range(n):
        left = max(i - locations[i], 0)
        right = min(i + locations[i], n - 1)
        ranges.append((left, right))
    
    # Sort the ranges by their right endpoints
    ranges.sort(key=lambda x: x[1])
    
    min_lights = 0
    i = 0
    while i < n:
        min_right = ranges[i][1]
        min_lights += 1
        while i < n and ranges[i][0] <= min_right:
            i += 1
    
    return min_lights

n = 7
locations = [0, 3, 0, 1, 0, 2, 0]
# print(min_lights_to_cover_road(n , locations))
'''

#########################
#   MAXIMIZE EARNINGS   #
#########################
'''
class Job:
    def __init__(self, st, ed, cost):
        self.st = st
        self.ed = ed
        self.cost = cost

class Pair:
    def __init__(self, first, second):
        self.first = first
        self.second = second

def search_job(arr, st, ed, key):
    ans = -1
    while st <= ed:
        mid = (st + ed) // 2
        if arr[mid].ed <= key:
            ans = mid
            st = mid + 1
        else:
            ed = mid - 1
    return ans

def get_time(st):
    hr = (int(st[0]) - 0) * 10 + (int(st[1]) - 0)
    min = (int(st[2]) - 0) * 10 + (int(st[3]) - 0)
    return hr * 60 + min


# HackerRank should provide these codes below:
n = int(input())
jobs = []

for _ in range(n):
    st = input()
    ed = input()
    cost = int(input())
    jobs.append(Job(get_time(st), get_time(ed), cost))

jobs.sort(key=lambda x: x.ed)
dp = [0] * n
num_of_jobs = [0] * n
dp[0] = jobs[0].cost
num_of_jobs[0] = 1

for i in range(1, n):
    idx = search_job(jobs, 0, i - 1, jobs[i].st)
    if idx != -1:
        curr = jobs[i].cost + dp[idx]
        num = num_of_jobs[idx] + 1
        if curr > dp[i - 1]:
            dp[i] = curr
            num_of_jobs[i] = num
        else:
            dp[i] = dp[i - 1]
            num_of_jobs[i] = num_of_jobs[i - 1]
    else:
        dp[i] = max(dp[i - 1], jobs[i].cost)
        num_of_jobs[i] = 1

result = Pair(num_of_jobs[n - 1], dp[n - 1])
selected_jobs_index = result.first
remaining_jobs = n - result.first
remaining_earnings = sum(jobs[i].cost for i in range(n) if i != selected_jobs_index)
print(remaining_jobs)
print(remaining_earnings)
'''


#########################
#    NETWORK STREAM     #
#########################

'''
def large_repackaged_packet(arr):
    max_power_of_two = 0
    x = 0

    for packet in arr:
        packet += x
        power_of_two = 0
        while packet >= 2 ** power_of_two:
            power_of_two += 1

        x = packet - 2 ** (power_of_two - 1)

        if max_power_of_two <= 2 ** (power_of_two - 1):
            max_power_of_two = 2 ** (power_of_two - 1)

    return max_power_of_two

# HackerRank should provide these codes below:
packets = [int(input()) for _ in range(int(input()))]
print(large_repackaged_packet(packets))
'''


#########################
#   ASTRONOMY LECTURE   #
#########################
'''
n = int(input())
for i in range(n):
    print("*" * (n - 1 - i) + "." * (2 * i + 1) + "*" * (n - 1 - i))
for i in range(n - 1):
    print("*" * (i + 1) + "." * (2 * (n - 2 - i) + 1) + "*" * (i + 1))
'''


#########################
#  DISK SPACE ANALYSIS  #
#########################
'''
def max_of_min_subarrays(s, n, a):
    max_of_mins = float('-inf')
    window = []

    for i in range(n):
        while window and a[i] < a[window[-1]]:
            window.pop()

        window.append(i)

        if i >= s - 1:
            while window and window[0] <= i - s:
                window.pop(0)

            max_of_mins = max(max_of_mins, a[window[0]])

    return max_of_mins

# HackerRank should provide these codes below:
s = int(input())
n = int(input())
a = []

for i in range(n):
    a.append(int(input()))

result = max_of_min_subarrays(s, n, a)
print(result)
'''


#########################
#     GUESS THE WORD    #
#########################
'''
# HackerRank should provide these codes below:
n = int(input())
sentence = input().split()
odd_lengths = [len(word) for word in sentence if len(word) % 2 == 1]

if not odd_lengths:
    print("Better luck next time")
else:
    max_length_word = sentence[odd_lengths.index(max(odd_lengths))]
    print(max_length_word)
'''


#########################
#  MINIMUM START VALUE  #
#########################

'''
n = int(input())
arr = [int(input()) for _ in range(n)]
s = a = 0

for i in arr:
    s += i
    if s < 1:
        a += (-s) + 1
        s = 1

print(a)
'''


#########################
#     COMPLEX MATH      #
#########################

'''
class ComplexNumber:
    def __init__(self, real, img):
        self.real = real
        self.img = img

    @staticmethod
    def add(c1, c2):
        return ComplexNumber(c1.real + c2.real, c1.img + c2.img)

a, b, c = map(int, input().split())

c1 = ComplexNumber(a, b)
print(f"{c1.real} + {c1.img}i")

c2 = ComplexNumber(c, 0)
c3 = ComplexNumber.add(c1, c2)
print(f"{c3.real} + {c3.img}i")

c4 = ComplexNumber.add(c3, c1)
print(f"{c4.real} + {c4.img}i")
'''


#########################
#     DEVIL GROUPS      #
#########################
'''
def devil_group(group):
    temp = ""
    result = []

    for person in group:
        temp += person
        if person == "$" or person == "@":
            result.append(temp)
            temp = ""

    result.append(temp)
    return len(max(result))

# print(devil_group("PPPPPP@PPP@PP$PPPPPP@PP"))
'''


#########################
#     VAMPIRE BATTLE    #
#########################
'''
def vampire_battle(people, power):
    curr_max = 0
    
    while sum(power) > curr_max:
        curr_max += max(power)
        power.remove(max(power))

    return curr_max

print(vampire_battle(6, [0,9,3,2,1,2,4,7]))
'''


#########################
#     COPYCAT EXAM      #
#########################
'''
def copycat_exam(paper_1, paper_2):
    sort_1, sort_2  = sorted(paper_1.lower()), sorted(paper_2.lower())

    if sort_1 == sort_2:
        return 0
    return 1

print(copycat_exam("CAR", "RAc"))
'''


###############################
#     MR. ROBOT's PASSWORD    #
###############################
'''
def robot_password(password):
    if len(password) < 6:
        return "Invalid Password"

    has_lowercase = any(c.islower() for c in password)
    has_uppercase = any(c.isupper() for c in password)
    has_numeric = any(c.isdigit() for c in password)

    if not (has_lowercase and has_uppercase and has_numeric):
        return "Invalid Password"

    if any(c == "/" or c == " " for c in password):
        return "Invalid Password"

    return "Valid Password"

print(robot_password("asdas234dL"))
'''


###############################
#       WEIRD TERMINAL        #
###############################
'''
def weird_terminal(sentence):
    temp = ""
    count = 0
    result = []

    for word in sentence:
        temp += word
        if word == " ":
            count += 1

        if count == 2:
            result.append(temp)
            count -= 2
            temp = ""

    result.append(temp)
    return len(result)

print(weird_terminal("How long do you have to sit dear?"))
'''


###############################
#      SET BIT CALCULATOR     #
###############################
'''
def set_bit_calc(nums):
    result = 0
    for n in nums:
        bin_form = bin(n)
        result += bin_form.count('1')
    return result

print(set_bit_calc([1,3,2,1]))
'''


###############################
#         EMAIL SPAM          #
###############################
'''
def spam_emails(mails):
    mail_count = {}
    total_spams = 0

    for m in mails:
        mail_count[m] = 1 + mail_count.get(m, 0)

    for count in mail_count.values():
        if count > 1:
            total_spams += count - 1

    return total_spams

print(spam_emails([1,1,3,3,4,3,3]))
'''


###############################
#     DEVICE NAME SYSTEM      #
###############################
'''
def device_name_sys(files):
    result = []
    dup_file = {}

    for file in files:
        if file in result:
            dup_file[file] = 1 + dup_file.get(file, 0)
            result.append(f"{file}{dup_file[file]}")
        else:
            result.append(file)

    return result

print(device_name_sys(["home","myFirst","dl","myFirst","myFirst"]))
'''


##################################
#    FORMATTING LARGE PRODUCTS   #
##################################
'''
from functools import lru_cache

@lru_cache(maxsize=None)
def format_large_products(a, b):
    result = 1

    for n in range(a, b+1):
        result = result * n

    return result

print(format_large_products(3,10))
'''


#######################
#     MAXIMUM TOYS    #
#######################
'''
def max_toys(money, toys):
    left = 0
    current_sum = 0
    max_toys_count = 0

    for right in range(len(toys)):
        current_sum += toys[right]

        while current_sum > money:
            current_sum -= toys[left]
            left += 1

        max_toys_count = max(max_toys_count, right - left + 1)

    return max_toys_count
print(max_toys(6, [1,4,5,3,2,1,6]))
'''


#############################
#     MAXIMUM ATTENDANCE    #
#############################
'''
def max_attendance(attendance):
    curr_sum = 0
    result = 0

    for data in attendance:
        if all(d == "P" for d in data):
            curr_sum += 1
            result = max(curr_sum, result)
        else:
            curr_sum = 0

    return result

print(max_attendance(["PPPP","PPPP","PPPP","PPAP","AAPP","PAPA","AAAA"]))
'''


#############################
#      SOLVE EQUATIONS      #
#############################
'''
# eval() method approach
def solve_equations(nums, equations):
    for equation in equations:
        for i in range(len(nums)):
            x = nums[i]
            nums[i] = eval(equation)

    return nums

# Custom dictionary approach
def solve_equations(nums, equations):
    ops_dict = {
        "x*x":lambda x: x * x,
        "x+x":lambda x: x + x,
        "3*x+x":lambda x: 3 * x + x
        }

    for i in range(len(nums)):
        for eq in equations:
            if eq in ops_dict:
                nums[i] = ops_dict[eq](nums[i])
    return nums    

print(solve_equations([2, 3, 4, 5, 1], ["x*x", "x+x", "3*x+x"]))
'''


#############################
#    MAXIMUM LIGHTHOUSE     #
#############################
'''
def max_lighthouses(n, positions):
    # Sort the positions in ascending order
    positions.sort()
    
    max_lighthouses = 0
    current_count = 1  # Initialize with the first lighthouse

    for i in range(1, n):
        if positions[i] - positions[i - 1] <= 1:
            current_count += 1
        else:
            max_lighthouses = max(max_lighthouses, current_count)
            current_count = 1  # Reset current count for the next sequence

    return max(max_lighthouses, current_count)

print(max_lighthouses(5, [4, 3, 2, 7, 8]))
'''


#################
#    MATCH      #
#################
'''
def count_matches(team_a, team_b):
    count = 0
    for a_wins, b_wins in zip(team_a, team_b):
        if a_wins <= b_wins:
            count += 1
    return count

print(count_matches([1, 1, 0], [0, 0, 1]))
'''


#####################
#    JUMBLE WORDS   #
#####################
'''
def jumble_words(word_a, word_b):
    result = ""
    min_length = min(len(word_a), len(word_b))

    for i in range(min_length):
        result += word_a[i] + word_b[i]

    # Add any remaining characters from the longer word, if any
    result += word_a[min_length:] + word_b[min_length:]

    return result

print(jumble_words("abc", "defksdfj"))
'''


#########################
#    REPEATED STRING    #
#########################

'''
# My Approach
def repeatedString(s, n):
    N = len(s)
    estimate = n // N
    remainder = n % N
    positions = []
    
    count = 0
    for i in range(N):
        if s[i] == "a":
            count += 1
            positions.append(i)
            
    result = count * estimate
            
    for pos in positions:
        if pos < remainder:
            result += 1

    return result

# ChatGPT's Approach
def repeatedString(s, n):
    count_in_single = s.count("a")
    estimate = n // len(s)
    remainder = n % len(s)
    count_in_remainder = s[:remainder].count("a")

    result = count_in_single * estimate + count_in_remainder

    return result    
'''


#########################
#    REPEATED STRING    #
#########################
'''
def utopian_tree(cycles):
    return (1 << ((cycles >> 1) + 1)) - 1 << cycles % 2

print(utopian_tree(3))
'''


#########################
#  DIAGONAL DIFFERENCE  #
#########################
'''
def diagonal_difference(matrix):
    main_diag_sum = 0
    secondary_diag_sum = 0
    n = len(matrix)

    for i in range(n):
        main_diag_sum += matrix[i][i]
        secondary_diag_sum += matrix[i][n - 1 - i]

    diff = abs(main_diag_sum - secondary_diag_sum)

    return diff
'''

# D@wnCh3ckCODES <-- Codequiry