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

