#! /usr/bin/python3.10

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import pdb
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd
import multiprocessing as mp

from collections import defaultdict
from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from hashlib import sha256
from io import BytesIO
from memory_tempfile import MemoryTempfile
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union, Any
from PIL import Image

CURRENT_WORKING_DIR = os.getcwd()
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = MemoryTempfile().gettempdir()

class Field:
	def __init__(self):
		pass


class FieldCell:
	def __init__(self, idx_y, idx_x, pos_y, pos_x, name_of_field):
		self.idx_y = idx_y
		self.idx_x = idx_x
		self.pos_y = pos_y
		self.pos_x = pos_x
		self.name_of_field = name_of_field
		self.d_player_color_to_d_next_field = {}
		self.d_player_color_to_field_cell_nr = {}
		# a field_cell can only contain one current_piece!
		self.current_piece = None


	# def __repr__(self):
	# 	return f"FieldCell(idx_y={self.idx_y}, idx_x={self.idx_x}, pos_y={self.pos_y}, pos_x={self.pos_x}, name_of_field='{self.name_of_field}', d_player_color_to_d_next_field={self.d_player_color_to_d_next_field}, d_player_color_to_field_cell_nr={self.d_player_color_to_field_cell_nr}, current_piece={self.current_piece})"
	#

	# the other field_cell must be empty, otherwise there will be a contradiction!
	def move_piece_to_next_field_cell(self, amount_move: int) -> bool:
		assert amount_move >= 1
		assert amount_move <= 6 # should be made with a const variable!
		assert not isinstance(self.current_piece, None)

		d_next_field = self.d_player_color_to_d_next_field[self.current_piece.color]

		if amount_move not in d_next_field:
			return False

		next_field_cell = d_next_field[amount_move]
		
		if isinstance(next_field_cell.current_piece, None):
			return False

		other.current_piece = self.current_piece
		self.current_piece = None

		other.current_piece.current_field_cell = other # update the current_field_cell of the found current_piece

		return True


class Piece:
	"""
	A playable piece of the game.

	...

	Attributes
	----------
	name : str
		the name of the piece e.g. 'red_1'
	color : str
		the color of the piece e.g. 'red'
	number : int
		the number of the piece e.g. 2
	d_tile_set : Dict[str, np.ndarray]
		the type of the tile, which will be used for active or non active piece

	Methods
	-------
	"""

	def __init__(
		self,
		name: str,
		color: str,
		number: int,
	):
		self.name = name
		self.color = color
		self.number = number
		self.current_field_cell = None
		self.home_field_cell = None
		self.state_active: bool = False
		self.state_finish: bool = False


def set_tile_into_field(pix_field, pix_tile, y, x):
	assert y >= 0 # is not implemented yet
	assert x >= 0 # is not implemented yet

	h_f, w_f, d_f = pix_field.shape
	h_t, w_t, d_t = pix_tile.shape

	assert d_f == 4 # RGBA is needed in deed!
	assert d_t == 4

	y_1 = min(y + h_t, h_f)
	x_1 = min(x + w_t, w_f)

	h_t_1 = y_1 - y if y_1 == h_f else h_t
	w_t_1 = x_1 - x if x_1 == w_f else w_t

	pix_tile_part = pix_tile[:h_t_1, :w_t_1]

	pix_tile_part_alpha = (pix_tile_part[:, :, 3].astype(np.float64) / 255)
	pix_tile_part_alpha_resh = pix_tile_part_alpha.reshape(pix_tile_part.shape[:2]).reshape(pix_tile_part.shape[:2] + (1, ))

	pix_field[y:y_1, x:x_1, :3] = ((1 - pix_tile_part_alpha_resh) * pix_field[y:y_1, x:x_1, :3] + pix_tile_part_alpha_resh * pix_tile_part[:, :, :3]).astype(np.uint8)
	pix_field[y:y_1, x:x_1, 3] = (((1 - pix_tile_part_alpha) * ((pix_field[y:y_1, x:x_1, 3].astype(np.float64) / 255)) + pix_tile_part_alpha) * 255.999999).astype(np.uint8)


if __name__ == '__main__':
	dir_img_path = os.path.join(CURRENT_WORKING_DIR, 'img')
	assert os.path.exists(dir_img_path)

	tile_size = 16
	tiles_amount_h = 11
	tiles_amount_w = 11

	# TODO: make this general for any tile_size in near future
	l_file_name = [
		'empty_16x16.png',
		'empty_frame_16x16.png',
		'red_16x16.png',
		'green_16x16.png',
		'yellow_16x16.png',
		'blue_16x16.png',
		'number_1_16x16.png',
		'number_2_16x16.png',
		'number_3_16x16.png',
		'number_4_16x16.png',
		'select_piece_16x16.png',
	]

	pix_h = tiles_amount_h * tile_size
	pix_w = tiles_amount_w * tile_size

	pix = np.zeros((pix_h, pix_w, 4), dtype=np.uint8)

	pix_empty = np.array(Image.open(os.path.join(dir_img_path, 'empty_16x16.png')))
	pix_empty_frame = np.array(Image.open(os.path.join(dir_img_path, 'empty_frame_16x16.png')))
	
	pix_red = np.array(Image.open(os.path.join(dir_img_path, 'red_16x16.png')))
	pix_green = np.array(Image.open(os.path.join(dir_img_path, 'green_16x16.png')))
	pix_yellow = np.array(Image.open(os.path.join(dir_img_path, 'yellow_16x16.png')))
	pix_blue = np.array(Image.open(os.path.join(dir_img_path, 'blue_16x16.png')))

	pix_number_1 = np.array(Image.open(os.path.join(dir_img_path, 'number_1_16x16.png')))
	pix_number_2 = np.array(Image.open(os.path.join(dir_img_path, 'number_2_16x16.png')))
	pix_number_3 = np.array(Image.open(os.path.join(dir_img_path, 'number_3_16x16.png')))
	pix_number_4 = np.array(Image.open(os.path.join(dir_img_path, 'number_4_16x16.png')))

	pix_empty_frame_red = np.array(Image.open(os.path.join(dir_img_path, 'empty_frame_red_16x16.png')))
	pix_empty_frame_green = np.array(Image.open(os.path.join(dir_img_path, 'empty_frame_green_16x16.png')))
	pix_empty_frame_yellow = np.array(Image.open(os.path.join(dir_img_path, 'empty_frame_yellow_16x16.png')))
	pix_empty_frame_blue = np.array(Image.open(os.path.join(dir_img_path, 'empty_frame_blue_16x16.png')))
	
	pix_arrow_right = np.array(Image.open(os.path.join(dir_img_path, 'arrow_right_16x32.png')))
	
	pix_select_piece = np.array(Image.open(os.path.join(dir_img_path, 'select_piece_16x16.png')))

	d_tile_name_to_pix = {
		'empty': pix_empty,
		'empty_frame': pix_empty_frame,
		'red': pix_red,
		'green': pix_green,
		'yellow': pix_yellow,
		'blue': pix_blue,
		'nr_1': pix_number_1,
		'nr_2': pix_number_2,
		'nr_3': pix_number_3,
		'nr_4': pix_number_4,
		'empty_frame_red': pix_empty_frame_red,
		'empty_frame_green': pix_empty_frame_green,
		'empty_frame_yellow': pix_empty_frame_yellow,
		'empty_frame_blue': pix_empty_frame_blue,
		'arrow_right': pix_arrow_right,
		'arrow_down': pix_arrow_right.transpose(1, 0, 2),
		'arrow_left': np.flip(pix_arrow_right, axis=1),
		'arrow_up': np.flip(pix_arrow_right, axis=1).transpose(1, 0, 2),
		'select_piece': pix_select_piece,
	}

	for y in range(0, pix_h, tile_size):
		for x in range(0, pix_w, tile_size):
			set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty'], y=y, x=x)

	# the inside empty_frame
	for i in range(0, 5):
		set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame'], y=4*tile_size, x=i*tile_size)
		set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame'], y=6*tile_size, x=i*tile_size)

		set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame'], y=4*tile_size, x=(6+i)*tile_size)
		set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame'], y=6*tile_size, x=(6+i)*tile_size)

		set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame'], x=4*tile_size, y=i*tile_size)
		set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame'], x=6*tile_size, y=i*tile_size)

		set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame'], x=4*tile_size, y=(6+i)*tile_size)
		set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame'], x=6*tile_size, y=(6+i)*tile_size)
	
	# the outside center empty_frame
	set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame'], y=0*tile_size, x=5*tile_size)
	set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame'], y=10*tile_size, x=5*tile_size)
	set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame'], y=5*tile_size, x=0*tile_size)
	set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame'], y=5*tile_size, x=10*tile_size)
	
	# the center empty_frame
	set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame'], y=5*tile_size, x=5*tile_size)

	# set the arrows
	set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['arrow_right'], y=3*tile_size, x=0*tile_size)
	set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['arrow_down'], y=0*tile_size, x=7*tile_size)
	set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['arrow_left'], y=7*tile_size, x=9*tile_size)
	set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['arrow_up'], y=9*tile_size, x=3*tile_size)

	# starting fields
	set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame_red'], y=4*tile_size, x=0*tile_size)
	set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame_green'], y=0*tile_size, x=6*tile_size)
	set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame_yellow'], y=6*tile_size, x=10*tile_size)
	set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame_blue'], y=10*tile_size, x=4*tile_size)

	# the goal empty_frame colors
	for i in range(0, 4):
		set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame_red'], y=5*tile_size, x=(1+i)*tile_size)
		set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame_green'], x=5*tile_size, y=(1+i)*tile_size)
		set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame_yellow'], y=5*tile_size, x=(5+1+i)*tile_size)
		set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix['empty_frame_blue'], x=5*tile_size, y=(5+1+i)*tile_size)

	color_player_1 = 'red'
	color_player_2 = 'green'
	color_player_3 = 'yellow'
	color_player_4 = 'blue'

	# tile_name, y, x
	l_tile_name_place = [
		('empty_frame', 0, 0),
		('empty_frame', 0, tile_size),
		('empty_frame', tile_size, 0),
		('empty_frame', tile_size, tile_size),

		('empty_frame', 0, 0+9*tile_size),
		('empty_frame', 0, tile_size+9*tile_size),
		('empty_frame', tile_size, 0+9*tile_size),
		('empty_frame', tile_size, tile_size+9*tile_size),

		('empty_frame', 0+9*tile_size, 0),
		('empty_frame', 0+9*tile_size, tile_size),
		('empty_frame', tile_size+9*tile_size, 0),
		('empty_frame', tile_size+9*tile_size, tile_size),

		('empty_frame', 0+9*tile_size, 0+9*tile_size),
		('empty_frame', 0+9*tile_size, tile_size+9*tile_size),
		('empty_frame', tile_size+9*tile_size, 0+9*tile_size),
		('empty_frame', tile_size+9*tile_size, tile_size+9*tile_size),

		(color_player_1, 0, 0),
		(color_player_1, 0, tile_size),
		(color_player_1, tile_size, 0),
		(color_player_1, tile_size, tile_size),

		(color_player_2, 0, 0+9*tile_size),
		(color_player_2, 0, tile_size+9*tile_size),
		(color_player_2, tile_size, 0+9*tile_size),
		(color_player_2, tile_size, tile_size+9*tile_size),

		(color_player_3, 0+9*tile_size, 0+9*tile_size),
		(color_player_3, 0+9*tile_size, tile_size+9*tile_size),
		(color_player_3, tile_size+9*tile_size, 0+9*tile_size),
		(color_player_3, tile_size+9*tile_size, tile_size+9*tile_size),

		(color_player_4, 0+9*tile_size, 0),
		(color_player_4, 0+9*tile_size, tile_size),
		(color_player_4, tile_size+9*tile_size, 0),
		(color_player_4, tile_size+9*tile_size, tile_size),

		('nr_1', 0, 0),
		('nr_2', 0, tile_size),
		('nr_3', tile_size, 0),
		('nr_4', tile_size, tile_size),

		('nr_1', 0, 0+9*tile_size),
		('nr_2', 0, tile_size+9*tile_size),
		('nr_3', tile_size, 0+9*tile_size),
		('nr_4', tile_size, tile_size+9*tile_size),

		('nr_1', 0+9*tile_size, 0),
		('nr_2', 0+9*tile_size, tile_size),
		('nr_3', tile_size+9*tile_size, 0),
		('nr_4', tile_size+9*tile_size, tile_size),

		('nr_1', 0+9*tile_size, 0+9*tile_size),
		('nr_2', 0+9*tile_size, tile_size+9*tile_size),
		('nr_3', tile_size+9*tile_size, 0+9*tile_size),
		('nr_4', tile_size+9*tile_size, tile_size+9*tile_size),

		('select_piece', 0, 0),
	]

	for tile_name, y, x in l_tile_name_place:
		set_tile_into_field(pix_field=pix, pix_tile=d_tile_name_to_pix[tile_name], y=y, x=x)

	# create all field_cell and link them up for the next step! plus the finish field_cell for each color

	# create a incrementely tuple list beginning from the (y, x) position (4, 0)
	l_pos_inc = (
		[(0, 1) for _ in range (0, 4)] +
		[(-1, 0) for _ in range (0, 4)] +
		[(0, 1) for _ in range (0, 2)] +

		[(1, 0) for _ in range (0, 4)] +
		[(0, 1) for _ in range (0, 4)] +
		[(1, 0) for _ in range (0, 2)] +

		[(0, -1) for _ in range (0, 4)] +
		[(1, 0) for _ in range (0, 4)] +
		[(0, -1) for _ in range (0, 2)] +

		[(-1, 0) for _ in range (0, 4)] +
		[(0, -1) for _ in range (0, 4)] +
		[(-1, 0) for _ in range (0, 2)]
	)

	l_pos_abs = []
	y_abs = 4
	x_abs = 0

	for y_inc, x_inc in l_pos_inc:
		y_abs += y_inc
		x_abs += x_inc
		l_pos_abs.append((y_abs, x_abs))

	# shift the list by one element right
	l_pos_abs = l_pos_abs[-1:] + l_pos_abs[:-1]

	l_player_color = ['red', 'green', 'yellow', 'blue']
	d_player_color_to_l_final_pos_inc = {
		'red': [(0, 1) for _ in range(0, 4)],
		'green': [(1, 0) for _ in range(0, 4)],
		'yellow': [(0, -1) for _ in range(0, 4)],
		'blue': [(-1, 0) for _ in range(0, 4)],
	}

	# create all possible field for each piece individually
	d_l_player_idx_pos_abs = {}
	for player_color in l_player_color:
		l_final_pos_inc = d_player_color_to_l_final_pos_inc[player_color]

		y_abs, x_abs = l_pos_abs[-1] # get the last pos to calc the last final fields too
		l_final_pos_abs = []
		for y_inc, x_inc in l_final_pos_inc:
			y_abs += y_inc
			x_abs += x_inc
			l_final_pos_abs.append((y_abs, x_abs))

		l_player_pos_abs = l_pos_abs + l_final_pos_abs
		d_l_player_idx_pos_abs[player_color] = l_player_pos_abs

		# shift the list by 10 elements left for the next piece positions
		l_pos_abs = l_pos_abs[10:] + l_pos_abs[:10]

	# and calculate a lookup table from each field_cell to the next possible field_cell, if the dice is a 6 sided one
	# e.g. from the position 2 the piece can go next to 3, 4, 5, 6, 7 and 8 because of all the possibilities

	d_player_color_to_d_lookup_table = {}
	for player_color in l_player_color:
		l_player_pos_abs = d_l_player_idx_pos_abs[player_color]

		d_lookup_table = {}
		for i in range(1, 6+1): # 6 sided dice
			d_lookup_table[i] = {pos_1: pos_2 for pos_1, pos_2 in zip(l_player_pos_abs[:-i], l_player_pos_abs[i:])}

		d_player_color_to_d_lookup_table[player_color] = d_lookup_table

	d_idx_pos_to_field_cell = {}
	# now set for all individual field_cell the next possible field_cells
	for player_color_1, player_color_2 in zip(l_player_color, l_player_color[1:]+l_player_color[:1]):
		l_player_pos_abs = d_l_player_idx_pos_abs[player_color_1]

		y, x = l_player_pos_abs[0] # the first entrance of the player_color_1
		d_idx_pos_to_field_cell[(y, x)] = FieldCell(
			idx_y=y, idx_x=x,
			pos_y=y*tile_size, pos_x=x*tile_size,
			name_of_field=f"entrance_{player_color_1}",
		)

		for i in range(1, 9):
			y, x = l_player_pos_abs[i] # the way of the player_color_1
			d_idx_pos_to_field_cell[(y, x)] = FieldCell(
				idx_y=y, idx_x=x,
				pos_y=y*tile_size, pos_x=x*tile_size,
				name_of_field=f"road_{player_color_1}_nr_{i}",
			)
		
		y, x = l_player_pos_abs[9] # the exit field_cell of player_color_2
		d_idx_pos_to_field_cell[(y, x)] = FieldCell(
			idx_y=y, idx_x=x,
			pos_y=y*tile_size, pos_x=x*tile_size,
			name_of_field=f"exit_{player_color_2}",
		)

		for i, (y, x) in enumerate(l_player_pos_abs[-4:], 1): # the last fields of the player_color_1
			d_idx_pos_to_field_cell[(y, x)] = FieldCell(
				idx_y=y, idx_x=x,
				pos_y=y*tile_size, pos_x=x*tile_size,
				name_of_field=f"finish_{player_color_1}_nr_{i}",
			)

	for player_color in l_player_color:
		l_player_pos_abs = d_l_player_idx_pos_abs[player_color]
		
		for field_cell_nr, pos_abs in enumerate(l_player_pos_abs, 1):
			field_cell = d_idx_pos_to_field_cell[pos_abs]
			field_cell.d_player_color_to_field_cell_nr[player_color] = field_cell_nr

		d_lookup_table = d_player_color_to_d_lookup_table[player_color]

		for amount_move, d_idx_pos_1_to_idx_pos_2 in d_lookup_table.items():
			for idx_pos_1, idx_pos_2 in d_idx_pos_1_to_idx_pos_2.items():
				field_cell_1 = d_idx_pos_to_field_cell[idx_pos_1]
				field_cell_2 = d_idx_pos_to_field_cell[idx_pos_2]

				if player_color not in field_cell_1.d_player_color_to_d_next_field:
					field_cell_1.d_player_color_to_d_next_field[player_color] = {}

				field_cell_1.d_player_color_to_d_next_field[player_color][amount_move] = field_cell_2				

	# create all the needed pieces
	d_player_color_to_d_piece = {}
	for player_color in l_player_color:
		d_piece = {}
		for number in range(1, 4+1):
			d_piece[number] = Piece(
				name=f'piece_{player_color}_nr_{number}',
				color=player_color,
				number=number,
			)

		d_player_color_to_d_piece[player_color] = d_piece

	# create the starting fields and associate the starting field with each piece too
	d_player_color_to_piece_home_field_cell_idx_pos = {
		'red': (0, 0),
		'green': (0, 9),
		'yellow': (9, 9),
		'blue': (9, 0),
	}

	d_player_color_to_d_starting_field_cell = {}
	for player_color in l_player_color:
		l_player_pos_abs = d_l_player_idx_pos_abs[player_color]
		starting_player_pos_abs = l_player_pos_abs[0]
		starting_field_cell = d_idx_pos_to_field_cell[starting_player_pos_abs]

		starting_piece_idx_pos = d_player_color_to_piece_home_field_cell_idx_pos[player_color]
		idx_y, idx_x = starting_piece_idx_pos

		d_starting_field_cell = {}

		piece_number = 1
		for y_inc in range(0, 2):
			idx_y_abs = idx_y + y_inc
			for x_inc in range(0, 2):
				idx_x_abs = idx_x + x_inc

				field_cell = FieldCell(
					idx_y=idx_y_abs, idx_x=idx_x_abs,
					pos_y=idx_y_abs*tile_size, pos_x=idx_x_abs*tile_size,
					name_of_field=f"starting_{player_color_1}_nr_{piece_number}",
				)
				field_cell.d_player_color_to_field_cell_nr[player_color] = 0
				field_cell.d_player_color_to_d_next_field[player_color] = {6: starting_field_cell} # TODO: the 6 should be made generic with a const variable!

				d_idx_pos_to_field_cell[(idx_y_abs, idx_x_abs)] = field_cell

				piece = d_player_color_to_d_piece[player_color][piece_number]
				
				piece.current_field_cell = field_cell
				piece.home_field_cell = field_cell
				field_cell.current_piece = piece

				d_starting_field_cell[(idx_y_abs, idx_x_abs)] = field_cell

				piece_number += 1

		d_player_color_to_d_starting_field_cell[player_color] = d_starting_field_cell

	# FIXME: change the priorities for each TODO if needed
	# TODO: for the ring fields (main field_cells only, not the home or the finish field_cells) for placing the piece back home, when 0 amount_move should be done!
	# TODO: make each move interactable with the updating of the playing field + log
	# TODO: make more functional style and oop style (do refactoring)
	# TODO: write function for the first interactions
	# TODO: make the most simplest rule for playing the game automatically
	# TODO: make possible for playing 1, 2, 3 and 4 players
	# TODO: write a log file for each game player until no more moves can be made!
	# TODO: write test cases later for testing the play field!

	img_test = Image.fromarray(pix)
	img_test.save(os.path.join(TEMP_DIR, 'test.png'))
