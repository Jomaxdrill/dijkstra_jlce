import heapq as hq
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import animation
from functools import partial
import numpy as np

####*DEFINE CONSTANTS #####
#used for the animation of solution
width_space = 1200
height_space = 500
frames_per_event = 1000
nodes_frames_min_event = 20
background_color = np.array([255, 255, 255], dtype=np.uint8)
obstacles_color = 'orange'
nodes_color = np.array([0, 255, 0], dtype=np.uint8)
initial_state_color = 'yellow'
goal_state_color = 'blue'
goal_path_color = np.array([255, 0, 0], dtype=np.uint8)
space = np.full((500,1200,3), background_color)
clearance = 5
action_operator = {
	'up': (0,1),
	'down': (0,-1),
	'left': (-1,0),
	'right': (1,0),
	'up-left':(1,-1),
	'up-right':(1,1),
	'down-left':(-1,-1),
	'down-right':(-1,1)
}
#####* INITIALIZATION OF MAIN VARIABLES #####
generated_nodes = [] #open list
#* Set class has given excellent performance for Repeated-state checking
visited_nodes = [] #full list node visited
visited_nodes_states = set() #close list
goal_path = [] #least cost path to reach goal
solution = True #solution flag
#variables to calculate time taken
start_time = None
end_time = None
elapsed_time = None
#variables construct actions and dimensions
action_set = tuple(action_operator.keys())
cost_one = action_set[0:4]
cost_square_two = action_set[4:]
hq.heapify(generated_nodes)

####* FUNCTIONS FOR APPLY DIJKSTRA ALGORITHM #####
def coordinate_input(name):
	""" This function is a terminal user input interface for supply coordinates

	Args:
		name (string): name of the coordinate

	Raises:
		NotExactElements: coordinate is not exactly of length needed
		CoordsNotValid: coordinate not valid
		Repeat: coordinate not confirmed

	Returns:
		_type_: _description_
	"""
	value_provided = False
	while not value_provided:
		print(f'Enter coordinates of the { name } state:\n')
		try:
			input_coord = input('First horizontal axis then vertical one separated by comma ( eg: 3,8 ).')
			#print(input_coord)
			coord_process = input_coord.split(',')
			coord_process = [ int(element) for element in coord_process ]
			#user provided more or less elements than allowed
			if len(coord_process) != 2:
				raise Exception('NotExactElements')
			#coordinate is not valid
			is_in_obstacle = check_in_obstacle(coord_process)
			#print(coord_process)
			if is_in_obstacle:
				raise Exception('CoordsNotValid')
			confirm = input('Confirm coordinate? (y/n): ')
			if confirm == 'y':
				print(f'The coordinate is: { coord_process }')
				return tuple(coord_process)
			else:
				print('*****Must write coordinate again****')
				raise Exception('Repeat')
		except ValueError as error:
			print(error)
			print(f'Invalid input for the coordinate. Could not convert to integer all values.')
		except Exception as err:
			args = err.args
			if 'NotExactElements' in args:
				print('Coordinate should have exactly two values. Please try again.')
			elif 'CoordsNotValid' in args:
				print('Coordinate is in obstacle or outside space. Please try again.')
			else:
				print(err)

def check_in_obstacle(state):
	"""
	This function is used to check if a given state is inside the obstacles or not.

	Args:
		state (tuple): A tuple containing the x and y coordinates of the state.

	Returns:
		bool: A boolean value indicating if the state is inside the obstacles or not.

	Raises:
		ValueError: If the input state is not a tuple.

	"""
	tl = clearance
	x_pos, y_pos = state
	in_obstacle = np.zeros(6, dtype=bool)
	#outside of space
	if x_pos < 0 or y_pos < 0:
		# print(f'outside of space')
		return True
	if x_pos > width_space or y_pos > height_space:
		# print(f'outside of space')
		return True
	#first obstacle
	in_obstacle[0] = ( x_pos >= 100-tl and x_pos <= 175+tl ) and (y_pos >= 100-tl and y_pos <= height_space)
	if in_obstacle[0]:
		# print(f'first obstacle rectangle detected')
		return True
	#second obstacle
	in_obstacle[1] = ( x_pos >= 275-tl and x_pos <= 350+tl ) and (y_pos >= 0 and y_pos <= 400+tl )
	if in_obstacle[1]:
		# print(f'second obstacle rectangle detected')
		return True
	#third obstacle
	slope = 15/26
	half_primitive = np.zeros(5, dtype=bool)
	half_primitive[0] = ( y_pos + slope*x_pos -469.23 ) >= 0
	half_primitive[1] = ( y_pos + slope*x_pos -781 ) <= 0
	half_primitive[2] = ( y_pos - slope*x_pos +280.77 ) >= 0
	half_primitive[3] = ( y_pos - slope*x_pos -30.94 ) <= 0
	half_primitive[4] = x_pos >= 520-tl and x_pos <= 780+tl
	in_obstacle[2] = half_primitive.all()
	if in_obstacle[2]:
		# print(f'third obstacle hexagon detected')
		return True
	#fourth obstacle
	polygon_1 = np.zeros(3, dtype=bool)
	polygon_1[0] = ( x_pos >= 900-tl and x_pos <= 1100+tl ) and ( y_pos >= 375-tl and y_pos <= 450+tl )
	polygon_1[1] = ( x_pos >= 1020-tl and x_pos <= 1100+tl ) and ( y_pos >= 125 and y_pos <= 375 )
	polygon_1[2] =  ( x_pos >= 900-tl and x_pos <= 1100+tl ) and (y_pos >= 50-tl and y_pos <= 125+tl  )
	in_obstacle[3] = any(polygon_1)
	if in_obstacle[3]:
		# print(f'fourth obstacle ] shape detected')
		return True
	#border wall 1
	polygon_2 = np.zeros(3, dtype=bool)
	polygon_2[0] = ( x_pos >= 0 and x_pos <= 100-tl ) and (y_pos >= height_space-tl and y_pos <= height_space)
	polygon_2[1] = ( x_pos >= 0 and x_pos <= 5 ) and (y_pos >= tl and y_pos <= height_space-tl)
	polygon_2[2] =  ( x_pos >= 0 and x_pos <= 275-tl ) and (y_pos >= 0 and  y_pos <= tl )
	in_obstacle[4] = any(polygon_2)
	if in_obstacle[4]:
		# print(f'walls left detected')
		return True
	#border wall 2
	polygon_3 = np.zeros(3, dtype=bool)
	polygon_3[0] = ( x_pos >= 175+tl and x_pos <= width_space ) and (y_pos >= height_space-tl and y_pos <= height_space)
	polygon_3[1] = ( x_pos >= width_space-tl and x_pos <= width_space ) and ( y_pos >= tl and y_pos <= height_space-tl)
	polygon_3[2] =  ( x_pos >= 350+tl and x_pos <= width_space ) and ( y_pos >= 0 and y_pos <= tl )
	in_obstacle[5] = any(polygon_3)
	if in_obstacle[5]:
		# print(f'walls right detected')
		return True
	return False

def update_cost(current_cost, action):
	"""
	This function updates the cost of a node based on the action taken.

	Args:
		current_cost (float): The current cost of the node.
		action (str): The action taken to reach the node.

	Returns:
		float: The updated cost of the node.

	"""
	if action in cost_one:
		return current_cost + 1
	if action in cost_square_two:
		return current_cost + 1.4
	return current_cost

def apply_action(state, type_action):
	"""
	This function applies an action to a given state.

	Args:
		state (tuple): The current state of the puzzle.
		type_action (str): The type of action to apply.

	Returns:
		tuple: The new state after applying the action.

	Raises:
		ValueError: If the input action is not valid.

	"""
	action_to_do = action_operator.get(type_action, None)
	if action_to_do is None:
		return None
	return tuple(coor1 + coor2 for coor1, coor2 in zip(state, action_to_do))

def action_move(current_node, action):
	"""
	This function creates and verfies if the movement is valid or not
	Args:
		current_node (Node): Node to move

	Returns:
		Node: new Node with new configuration and state
	"""
	state_moved = apply_action(current_node[3], action)
	#print(state_moved)
	#*check new node is in obstacle space
	if check_in_obstacle(state_moved):
		#print(f'obstacle reached for { state_moved }')
		return None
	#*check by the state duplicate values between the children
	node_already_visited = state_moved in visited_nodes_states
	if node_already_visited:
		return None
	new_node = ( update_cost(current_node[0], action), -1, current_node[1], state_moved )
	return new_node

#?(cost, index, parent, state)
def create_nodes(initial_state, goal_state):
	"""Creates the State space of all possible movements until goal state is reached.

	Args:
		initial_state (array): multi dimensional array 3x3 that describes the initial configuarion of the puzzle
		goal_state (array): multi dimensional array 3x3 that describes the final configuration the algorithm must find.

	Returns:
		str: 'DONE'. The process have ended thus we have a solution in the tree structure generated.
	"""
	# Start the timer
	start_time = time.time()
	goal_reached = False
	counter_nodes = 0
	# Add initial node to the heap
	hq.heappush(generated_nodes, (0, counter_nodes, None, initial_state))
	while (not goal_reached) and len(generated_nodes):
		print(counter_nodes)
		# Remove the lowest cost path from the heap and store it into a variable
		current_node = generated_nodes[0]
		hq.heappop(generated_nodes)
		# For updating the heap structure
		hq.heapify(generated_nodes)
		# Mark node as visited
		visited_nodes.append(current_node)
		visited_nodes_states.add(current_node[3])
		# Check if popup_node is goal state
		goal_reached = current_node[3] == goal_state
		if goal_reached:
			goal_reached = True
			end_time = time.time()
			return f'DONE in {end_time-start_time} seconds.'
		#Apply action set to node to get new states/children
		for action in action_set:
			child = action_move(current_node, action)
			# If movement was not possible, ignore it
			if not child:
				continue
			# Check if child is in open list generated nodes
			where_is_node = 0
			is_in_open_list = False
			for node in generated_nodes:
				if node[3] == child[3]:
					is_in_open_list = True
					break
				where_is_node +=1
			if not is_in_open_list:
				counter_nodes += 1
				child_to_enter = (child[0],counter_nodes,child[2],child[3])
				hq.heappush(generated_nodes, child_to_enter)
			#check if cost is greater in node in open list
			elif generated_nodes[where_is_node][0] > child[0]:
				# Update parent node and cost of this child in the generated nodes heap
				generated_nodes[where_is_node] = (child[0], generated_nodes[where_is_node][1], child[2], child[3])
			# For updating the heap structure
			hq.heapify(generated_nodes)
	return False
def generate_path(node):
	"""Generate the path from the initial node to the goal state.

	Args:
		node (Node): Current node to evaluate its parent (previous move done).
	Returns:
		Boolean: True if no more of the path are available
	"""
	while node is not None:
		goal_path.append(node[3])
		parent_at = 0
		for node_check in visited_nodes:
			if node_check[1] == node[2]:
				break
			parent_at += 1
		node = visited_nodes[parent_at] if parent_at < len(visited_nodes) else None
	return True

###* FUNCTIONS FOR ANIMATE THE SOLUTION #####
def draw_obstacles():
	"""
	This function draws the obstacles in the space.
	"""
	obstacle_1 = plt.Rectangle((100,100), 75, 400, fc= obstacles_color)
	obstacle_2 = plt.Rectangle((275,0), 75, 400, fc= obstacles_color)
	points_obs_3 = ((650,400),(780,325),(780,175),(650,100),(520,175),(520,325))
	obstacle_3 = plt.Polygon(points_obs_3, fc= obstacles_color)
	obstacle_4_1 = plt.Rectangle((900,50), 200,75, fc= obstacles_color)
	obstacle_4_2 = plt.Rectangle((1020,125), 80, 250, fc= obstacles_color)
	obstacle_4_3 = plt.Rectangle((900,375), 200,75, fc= obstacles_color)
	circle_initial = plt.Circle(initial_state, 7.5, fc= initial_state_color)
	circle_goal = plt.Circle(goal_state, 7.5, fc= goal_state_color)
	set_figures = [ obstacle_1,
					obstacle_2,
					obstacle_3,
					obstacle_4_1,
					obstacle_4_2,
					obstacle_4_3,
					circle_initial,circle_goal ]
	for obstacle in set_figures:
		#* by patches the figures will remain as constat an unmutable in the plot
		axis.add_patch(obstacle)


def display(frame,chunks_nodes, chunks_goal_path):
	""" This function is the callback for the animation process

	Args:
		frame (int): current index of frame to display
		chunks_nodes (array): nodes per frame to display
		chunks_goal_path (array): nodes of the goal path to display

	Returns:
		array: iterator for the scene
	"""
	color_of_node = None
	chunk = None
	print(f' processing frame {frame}')
	if frame < len(chunks_nodes):
		chunk = chunks_nodes[frame]
		#print(f'chunk for visited node is {chunk}')
		color_of_node = nodes_color
		for value in chunk:
			col,row = value[3]
			space[row, col] = color_of_node
	elif frame < (len(chunks_nodes) + len(chunks_goal_path)):
		chunk = chunks_goal_path[frame-len(chunks_nodes)]
		color_of_node = goal_path_color
		#print(f'chunk for goal is {chunk}')
		for col,row in chunk:
			space[row, col] = color_of_node
	#* positions were given in (x,y) but the x is the column position so the y the row position
	space_image.set_array(space)
	return [space_image]


def processing_frames(arr_nodes):
	"""
	This function is used to calculate the number of nodes to process per frame.

	Args:
		arr_nodes (list): A list of nodes to process.

	Returns:
		list: A list of nodes to process per frame.

	"""
	# Calculate the number of nodes to process per frame
	total_nodes = len(arr_nodes)
	arr_per_frame =  total_nodes / frames_per_event
	print(f'points per frame should be {arr_per_frame}')
	# # If there are fewer nodes than the desired frames per event, process all nodes in one frame
	if arr_per_frame < 1 or (arr_per_frame >=1 and arr_per_frame <= nodes_frames_min_event):
		arr_per_frame = nodes_frames_min_event
	arr_per_frame = round(arr_per_frame)
	print(f'nodes per frame are {arr_per_frame}')
	chunks_arr = divide_array(arr_per_frame, arr_nodes)
	return chunks_arr

def divide_array(nodes_per_frame, arr_nodes):
	"""
	This function is used to divide an array into chunks of a specified size.

	Args:
		nodes_per_frame (int): The number of nodes to include in each chunk.
		arr_nodes (list): A list of nodes to divide.

	Returns:
		list: A list of lists, where each sub-list represents a chunk of nodes.

	"""
	arr_size = len(arr_nodes)
	if arr_size <= nodes_per_frame:
			return [ arr_nodes ]
	# Calculate the number of full chunks and the size of the remaining chunk
	number_full_slices  = arr_size // nodes_per_frame
	remaining_slice = arr_size % nodes_per_frame
	# Slice the array into chunks of the nodes per frame
	sliced_chunks = [ arr_nodes[idx*nodes_per_frame:(idx+1)*nodes_per_frame]
				for idx in range(number_full_slices) ]
	# Remaining nodes into a separate chunk
	if remaining_slice > 0:
		sliced_chunks.append(arr_nodes[number_full_slices*nodes_per_frame:])
	return sliced_chunks


######*INPUTS AND SOLUTIONS #####
print("-----------WELCOME TO PATH PLANNING SOLVER DIJKSTRA ALGORITHM-------------")
initial_state = coordinate_input('initial')
goal_state = coordinate_input('goal')
print(create_nodes(initial_state, goal_state))
print(f'total generated nodes were:{len(generated_nodes)}')
print(f'total visited nodes were:{len(visited_nodes)}')
generate_path(visited_nodes[-1])
print(f'total goal path is :{len(goal_path)}')
#######* GENERATE VIDEO #################################
print("-----------CREATING SPACE AND OBSTACLES-------------")
fig, axis = plt.subplots()
tl = clearance
draw_obstacles()
space_image = axis.imshow(space, origin='lower')
# plt.imshow(space, origin='lower')
# plt.show()
print(space_image)
