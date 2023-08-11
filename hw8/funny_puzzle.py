import heapq


def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    for i in range(len(from_state)):

        tile = from_state[i]
        if tile == 0:
            continue
        # find current position
        x_position = i % 3
        y_position = i // 3

        for j in range(len(to_state)):
            if to_state[j] == tile:

                # find the goal position
                x_goal = j % 3
                y_goal = j // 3


                distance += (abs(x_goal - x_position) + abs(y_goal - y_position))

    return distance




def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below).
    """

    succ_states = []

    for i in range(len(state)):
        if state[i] == 0:
            # find all the possible values that could fill in the 'empty space'
            if i % 3 != 0: left = state[i - 1]
            else: left = 0

            if i % 3 != 2: right = state[i + 1]
            else: right = 0

            if i < 6: down = state[i + 3]
            else: down = 0

            if i > 2: up = state[i - 3]
            else: up = 0

            # find and add the states after that move was completed
            if left != 0:
                move_right = state.copy()
                move_right[i - 1] = 0
                move_right[i] = left
                succ_states.append(move_right)

            if right != 0:
                move_left = state.copy()
                move_left[i + 1] = 0
                move_left[i] = right
                succ_states.append(move_left)

            if down != 0:
                move_up = state.copy()
                move_up[i + 3] = 0
                move_up[i] = down
                succ_states.append(move_up)

            if up != 0:
                move_down = state.copy()
                move_down[i - 3] = 0
                move_down[i] = up
                succ_states.append(move_down)
   
    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    visited_pq_entries = []
    visited_states = []
    pq = []
    parent_index = 0
    max_queue_length = 0
    g = 0
    h = get_manhattan_distance(state)
    heapq.heappush(pq,(g + h, state, (g, h, -1)))
    final_element = None

    while pq:
        current_element = heapq.heappop(pq)
        visited_pq_entries.append(current_element)
        visited_states.append(current_element[1])
        succ_states = get_succ(current_element[1])

        # exit when we reached the goal state
        if current_element[2][1] == 0:
            final_element = current_element
            break

        # add all the successor states to the queue that have not already been visited
        for succ_state in succ_states:
            if succ_state in visited_states:
                continue
            g = current_element[2][0] + 1
            h = get_manhattan_distance(succ_state)
            heapq.heappush(pq, (g + h, succ_state, (g, h, parent_index)))

        max_queue_length = max(max_queue_length, len(pq))

        parent_index += 1

    # create the solution path
    solution = []
    last_element = final_element
    while last_element[2][2] != -1:
        solution.append(last_element)
        next_element = visited_pq_entries[last_element[2][2]]
        last_element = next_element

    solution.append(last_element)


    count = 0
    for move in solution[::-1]:
        print(f'{move[1]} h={move[2][1]} moves: {count}')
        count += 1

    print(f'Max queue length: {max_queue_length}')








if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """


