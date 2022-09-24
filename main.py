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

def pix_alpha_blending(pix_dst, pix_src, y, x):
	assert y >= 0 # is not implemented yet
	assert x >= 0 # is not implemented yet

	h_f, w_f, d_f = pix_dst.shape
	h_t, w_t, d_t = pix_src.shape

	assert d_f == 4 # RGBA is needed in deed!
	assert d_t == 4

	y_1 = min(y + h_t, h_f)
	x_1 = min(x + w_t, w_f)

	h_t_1 = y_1 - y if y_1 == h_f else h_t
	w_t_1 = x_1 - x if x_1 == w_f else w_t

	pix_tile_part = pix_src[:h_t_1, :w_t_1]

	pix_tile_part_alpha = (pix_tile_part[:, :, 3].astype(np.float64) / 255)
	pix_tile_part_alpha_resh = pix_tile_part_alpha.reshape(pix_tile_part.shape[:2]).reshape(pix_tile_part.shape[:2] + (1, ))

	pix_dst[y:y_1, x:x_1, :3] = ((1 - pix_tile_part_alpha_resh) * pix_dst[y:y_1, x:x_1, :3] + pix_tile_part_alpha_resh * pix_tile_part[:, :, :3]).astype(np.uint8)
	pix_dst[y:y_1, x:x_1, 3] = (((1 - pix_tile_part_alpha) * ((pix_dst[y:y_1, x:x_1, 3].astype(np.float64) / 255)) + pix_tile_part_alpha) * 255.999999).astype(np.uint8)


def pix_alpha_blending_many(pix_dst, l_pix_src, l_y, l_x):
	for pix_src, y, x in zip(l_pix_src, l_y, l_x):
		pix_alpha_blending(pix_dst=pix_dst, pix_src=pix_src, y=y, x=x)


class GameField:
	def __init__(self, h, w):
		self.pix = np.zeros((h, w, 4), dtype=np.uint8)
		self.play_field_number = 0
		self.temp_dir_path = os.path.join(os.path.join(TEMP_DIR, 'man_do_not_get_angry'), datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H:%M:%S:%f'))
		if not os.path.exists(self.temp_dir_path):
			os.makedirs(self.temp_dir_path)


	def save_next_field_image(self):
		Image.fromarray(self.pix).save(os.path.join(self.temp_dir_path, f'field_nr_{self.play_field_number:03}.png'))
		self.play_field_number += 1


	def move_piece_from_starting_field_cell(self, piece: 'Piece', amount_move: int):
		current_field_cell = piece.current_field_cell
		other_piece, is_moved = current_field_cell.move_piece_to_next_field_cell(amount_move=amount_move)
		assert is_moved
		next_field_cell = piece.current_field_cell

		assert (other_piece is None) or (other_piece is not None and other_piece.color != piece.color)

		pos_y_1 = current_field_cell.pos_y
		pos_x_1 = current_field_cell.pos_x
		pos_y_2 = next_field_cell.pos_y
		pos_x_2 = next_field_cell.pos_x

		piece.state_select = True
		pix_alpha_blending(pix_dst=self.pix, pix_src=current_field_cell.d_tile_set["empty_cell"], y=pos_y_1, x=pos_x_1)
		if piece.state_select:
			pix_alpha_blending(pix_dst=self.pix, pix_src=piece.d_tile_set["select"], y=pos_y_1, x=pos_x_1)
		else:
			pix_alpha_blending(pix_dst=self.pix, pix_src=piece.d_tile_set["non_select"], y=pos_y_1, x=pos_x_1)
		self.save_next_field_image()
		
		pix_alpha_blending(pix_dst=self.pix, pix_src=current_field_cell.d_tile_set["empty_cell"], y=pos_y_1, x=pos_x_1)
		pix_alpha_blending(pix_dst=self.pix, pix_src=next_field_cell.d_tile_set["empty_cell"], y=pos_y_2, x=pos_x_2)
		if piece.state_select:
			pix_alpha_blending(pix_dst=self.pix, pix_src=piece.d_tile_set["select"], y=pos_y_2, x=pos_x_2)
		else:
			pix_alpha_blending(pix_dst=self.pix, pix_src=piece.d_tile_set["non_select"], y=pos_y_2, x=pos_x_2)
		if other_piece is not None:
			home_field_cell = other_piece.home_field_cell
			pos_y_3 = home_field_cell.pos_y
			pos_x_3 = home_field_cell.pos_x
			pix_alpha_blending(pix_dst=self.pix, pix_src=other_piece.d_tile_set["non_select"], y=pos_y_3, x=pos_x_3)
		self.save_next_field_image()

		piece.state_select = False
		pix_alpha_blending(pix_dst=self.pix, pix_src=next_field_cell.d_tile_set["empty_cell"], y=pos_y_2, x=pos_x_2)
		if piece.state_select:
			pix_alpha_blending(pix_dst=self.pix, pix_src=piece.d_tile_set["select"], y=pos_y_2, x=pos_x_2)
		else:
			pix_alpha_blending(pix_dst=self.pix, pix_src=piece.d_tile_set["non_select"], y=pos_y_2, x=pos_x_2)
		self.save_next_field_image()

		return True



class FieldCell:
	def __init__(self, idx_y, idx_x, pos_y, pos_x, name_of_field, is_starting_cell, is_entrance_cell, is_finish_cell):
		self.idx_y = idx_y
		self.idx_x = idx_x
		self.pos_y = pos_y
		self.pos_x = pos_x
		self.name_of_field = name_of_field
		self.is_starting_cell = is_starting_cell
		self.is_entrance_cell = is_entrance_cell
		self.is_finish_cell = is_finish_cell
		self.d_player_color_to_d_next_field = {}
		self.d_player_color_to_field_cell_nr = {}
		# a field_cell can only contain one current_piece!
		self.current_piece = None
		self.d_tile_set = {
			"empty_cell" : None,
		}


	# def __repr__(self):
	# 	return f"FieldCell(idx_y={self.idx_y}, idx_x={self.idx_x}, pos_y={self.pos_y}, pos_x={self.pos_x}, name_of_field='{self.name_of_field}', d_player_color_to_d_next_field={self.d_player_color_to_d_next_field}, d_player_color_to_field_cell_nr={self.d_player_color_to_field_cell_nr}, current_piece={self.current_piece})"
	#

	# the other field_cell must be empty, otherwise there will be a contradiction!
	def move_piece_to_next_field_cell(self, amount_move: int) -> bool:
		assert amount_move >= 1
		assert amount_move <= 6 # should be made with a const variable!
		current_piece = self.current_piece
		assert current_piece is not None

		d_next_field = self.d_player_color_to_d_next_field[current_piece.color]

		if amount_move not in d_next_field:
			return None, False

		next_field_cell = d_next_field[amount_move]
		
		other_piece = next_field_cell.current_piece
		if other_piece is not None:
			if other_piece.color == current_piece.color:
				return other_piece, False

			if current_piece.state_start:
				assert not other_piece.state_start
				current_piece.state_start = False

			other_piece.current_field_cell = other_piece.home_field_cell
			other_piece.home_field_cell.current_piece = other_piece
				
			other_piece.state_start = True

		current_piece.current_field_cell = next_field_cell # update the current_field_cell of the found current_piece

		next_field_cell.current_piece = current_piece
		self.current_piece = None

		return other_piece, True


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
		self.state_select: bool = False # the piece is selected right now
		self.state_start: bool = True # the piece is at the start
		self.state_finish: bool = False # the piece is safe in the finish
		self.state_finish_final: bool = False # absolutely no more moves can be made
		self.d_tile_set = {
			"select": None,
			"non_select": None,
		}


	def check_if_next_move_is_possible(self, amount_move):
		d_next_field = self.current_field_cell.d_player_color_to_d_next_field[self.color]

		if amount_move not in d_next_field:
			return False

		next_field_cell = d_next_field[amount_move]

		if next_field_cell.current_piece is not None and next_field_cell.current_piece.color == self.color:
			return False

		return True


if __name__ == '__main__':
	dir_img_path = os.path.join(CURRENT_WORKING_DIR, 'img')
	assert os.path.exists(dir_img_path)

	amount_player = 4
	amount_piece = 4

	tile_size = 16
	tiles_amount_h = amount_piece*2 + 3
	tiles_amount_w = 14

	# TODO: make this general for any tile_size in near future
	# l_file_name = [
	# 	'empty_16x16.png',
	# 	'empty_frame_16x16.png',
	# 	'red_16x16.png',
	# 	'green_16x16.png',
	# 	'yellow_16x16.png',
	# 	'blue_16x16.png',
	# 	'number_1_16x16.png',
	# 	'number_2_16x16.png',
	# 	'number_3_16x16.png',
	# 	'number_4_16x16.png',
	# 	'select_piece_16x16.png',
	# ]

	pix_h = tiles_amount_h * tile_size
	pix_w = tiles_amount_w * tile_size

	game_field = GameField(h=pix_h, w=pix_w)

	# game_field.pix = np.zeros((pix_h, pix_w, 4), dtype=np.uint8)

	pix_empty = np.array(Image.open(os.path.join(dir_img_path, 'empty_16x16.png')))
	pix_empty_frame = np.array(Image.open(os.path.join(dir_img_path, 'empty_frame_16x16.png')))
	pix_empty_mask = np.array(Image.open(os.path.join(dir_img_path, 'empty_mask_16x16.png')))
	
	pix_piece_red = np.array(Image.open(os.path.join(dir_img_path, 'piece_red_16x16.png')))
	pix_piece_green = np.array(Image.open(os.path.join(dir_img_path, 'piece_green_16x16.png')))
	pix_piece_yellow = np.array(Image.open(os.path.join(dir_img_path, 'piece_yellow_16x16.png')))
	pix_piece_blue = np.array(Image.open(os.path.join(dir_img_path, 'piece_blue_16x16.png')))

	pix_number_1 = np.array(Image.open(os.path.join(dir_img_path, 'number_1_16x16.png')))
	pix_number_2 = np.array(Image.open(os.path.join(dir_img_path, 'number_2_16x16.png')))
	pix_number_3 = np.array(Image.open(os.path.join(dir_img_path, 'number_3_16x16.png')))
	pix_number_4 = np.array(Image.open(os.path.join(dir_img_path, 'number_4_16x16.png')))
	
	pix_dice_number_0 = np.array(Image.open(os.path.join(dir_img_path, 'dice_nr_0_16x16.png')))
	pix_dice_number_1 = np.array(Image.open(os.path.join(dir_img_path, 'dice_nr_1_16x16.png')))
	pix_dice_number_2 = np.array(Image.open(os.path.join(dir_img_path, 'dice_nr_2_16x16.png')))
	pix_dice_number_3 = np.array(Image.open(os.path.join(dir_img_path, 'dice_nr_3_16x16.png')))
	pix_dice_number_4 = np.array(Image.open(os.path.join(dir_img_path, 'dice_nr_4_16x16.png')))
	pix_dice_number_5 = np.array(Image.open(os.path.join(dir_img_path, 'dice_nr_5_16x16.png')))
	pix_dice_number_6 = np.array(Image.open(os.path.join(dir_img_path, 'dice_nr_6_16x16.png')))

	pix_empty_frame_red = np.array(Image.open(os.path.join(dir_img_path, 'empty_frame_red_16x16.png')))
	pix_empty_frame_green = np.array(Image.open(os.path.join(dir_img_path, 'empty_frame_green_16x16.png')))
	pix_empty_frame_yellow = np.array(Image.open(os.path.join(dir_img_path, 'empty_frame_yellow_16x16.png')))
	pix_empty_frame_blue = np.array(Image.open(os.path.join(dir_img_path, 'empty_frame_blue_16x16.png')))
	
	pix_arrow_right = np.array(Image.open(os.path.join(dir_img_path, 'arrow_right_16x32.png')))
	
	pix_select_piece = np.array(Image.open(os.path.join(dir_img_path, 'select_piece_16x16.png')))

	d_tile_name_to_pix = {
		'empty': pix_empty,
		'empty_frame': pix_empty_frame,
		'empty_mask': pix_empty_mask,
		'piece_red': pix_piece_red,
		'piece_green': pix_piece_green,
		'piece_yellow': pix_piece_yellow,
		'piece_blue': pix_piece_blue,
		'nr_1': pix_number_1,
		'nr_2': pix_number_2,
		'nr_3': pix_number_3,
		'nr_4': pix_number_4,
		'dice_nr_0': pix_dice_number_0,
		'dice_nr_1': pix_dice_number_1,
		'dice_nr_2': pix_dice_number_2,
		'dice_nr_3': pix_dice_number_3,
		'dice_nr_4': pix_dice_number_4,
		'dice_nr_5': pix_dice_number_5,
		'dice_nr_6': pix_dice_number_6,
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
			pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty'], y=y, x=x)

	# the inside empty_frame
	for i in range(0, 5):
		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame'], y=4*tile_size, x=i*tile_size)
		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame'], y=6*tile_size, x=i*tile_size)

		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame'], y=4*tile_size, x=(6+i)*tile_size)
		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame'], y=6*tile_size, x=(6+i)*tile_size)

		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame'], x=4*tile_size, y=i*tile_size)
		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame'], x=6*tile_size, y=i*tile_size)

		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame'], x=4*tile_size, y=(6+i)*tile_size)
		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame'], x=6*tile_size, y=(6+i)*tile_size)
	
	# the outside center empty_frame
	pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame'], y=0*tile_size, x=5*tile_size)
	pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame'], y=10*tile_size, x=5*tile_size)
	pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame'], y=5*tile_size, x=0*tile_size)
	pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame'], y=5*tile_size, x=10*tile_size)
	
	# the center empty_frame
	pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame'], y=5*tile_size, x=5*tile_size)

	# set the arrows
	pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['arrow_right'], y=3*tile_size, x=0*tile_size)
	pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['arrow_down'], y=0*tile_size, x=7*tile_size)
	pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['arrow_left'], y=7*tile_size, x=9*tile_size)
	pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['arrow_up'], y=9*tile_size, x=3*tile_size)

	# starting fields
	pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame_red'], y=4*tile_size, x=0*tile_size)
	pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame_green'], y=0*tile_size, x=6*tile_size)
	pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame_yellow'], y=6*tile_size, x=10*tile_size)
	pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame_blue'], y=10*tile_size, x=4*tile_size)

	# the goal empty_frame colors
	for i in range(0, 4):
		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame_red'], y=5*tile_size, x=(1+i)*tile_size)
		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame_green'], x=5*tile_size, y=(1+i)*tile_size)
		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame_yellow'], y=5*tile_size, x=(5+1+i)*tile_size)
		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix['empty_frame_blue'], x=5*tile_size, y=(5+1+i)*tile_size)

	color_piece_1 = 'piece_red'
	color_piece_2 = 'piece_green'
	color_piece_3 = 'piece_yellow'
	color_piece_4 = 'piece_blue'

	# tile_name, y, x
	l_tile_name_place = [
		# ('empty_frame', 0, 0),
		# ('empty_frame', 0, tile_size),
		# ('empty_frame', tile_size, 0),
		# ('empty_frame', tile_size, tile_size),

		# ('empty_frame', 0, 0+9*tile_size),
		# ('empty_frame', 0, tile_size+9*tile_size),
		# ('empty_frame', tile_size, 0+9*tile_size),
		# ('empty_frame', tile_size, tile_size+9*tile_size),

		# ('empty_frame', 0+9*tile_size, 0),
		# ('empty_frame', 0+9*tile_size, tile_size),
		# ('empty_frame', tile_size+9*tile_size, 0),
		# ('empty_frame', tile_size+9*tile_size, tile_size),

		# ('empty_frame', 0+9*tile_size, 0+9*tile_size),
		# ('empty_frame', 0+9*tile_size, tile_size+9*tile_size),
		# ('empty_frame', tile_size+9*tile_size, 0+9*tile_size),
		# ('empty_frame', tile_size+9*tile_size, tile_size+9*tile_size),

		# (color_piece_1, 0, 0),
		# (color_piece_1, 0, tile_size),
		# (color_piece_1, tile_size, 0),
		# (color_piece_1, tile_size, tile_size),

		# (color_piece_2, 0, 0+9*tile_size),
		# (color_piece_2, 0, tile_size+9*tile_size),
		# (color_piece_2, tile_size, 0+9*tile_size),
		# (color_piece_2, tile_size, tile_size+9*tile_size),

		# (color_piece_3, 0+9*tile_size, 0+9*tile_size),
		# (color_piece_3, 0+9*tile_size, tile_size+9*tile_size),
		# (color_piece_3, tile_size+9*tile_size, 0+9*tile_size),
		# (color_piece_3, tile_size+9*tile_size, tile_size+9*tile_size),

		# (color_piece_4, 0+9*tile_size, 0),
		# (color_piece_4, 0+9*tile_size, tile_size),
		# (color_piece_4, tile_size+9*tile_size, 0),
		# (color_piece_4, tile_size+9*tile_size, tile_size),

		# ('nr_1', 0, 0),
		# ('nr_2', 0, tile_size),
		# ('nr_3', tile_size, 0),
		# ('nr_4', tile_size, tile_size),

		# ('nr_1', 0, 0+9*tile_size),
		# ('nr_2', 0, tile_size+9*tile_size),
		# ('nr_3', tile_size, 0+9*tile_size),
		# ('nr_4', tile_size, tile_size+9*tile_size),

		# ('nr_1', 0+9*tile_size, 0),
		# ('nr_2', 0+9*tile_size, tile_size),
		# ('nr_3', tile_size+9*tile_size, 0),
		# ('nr_4', tile_size+9*tile_size, tile_size),

		# ('nr_1', 0+9*tile_size, 0+9*tile_size),
		# ('nr_2', 0+9*tile_size, tile_size+9*tile_size),
		# ('nr_3', tile_size+9*tile_size, 0+9*tile_size),
		# ('nr_4', tile_size+9*tile_size, tile_size+9*tile_size),
		
		('dice_nr_0', 5*tile_size, 12*tile_size),
	]

	for tile_name, y, x in l_tile_name_place:
		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix[tile_name], y=y, x=x)

	game_field.save_next_field_image()
	
	# # tile_name, y, x
	# l_tile_name_place = [
	# 	('select_piece', 0, 0),
	# ]

	# for tile_name, y, x in l_tile_name_place:
	# 	pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix[tile_name], y=y, x=x)

	# Image.fromarray(game_field.pix).save(os.path.join(TEMP_DIR, 'man_do_not_get_angry_field_nr_01.png'))


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

	l_player_nr = [1, 2, 3, 4]
	l_player_color = ['red', 'green', 'yellow', 'blue']
	d_player_color_to_l_final_pos_inc = {
		'red': [(0, 1) for _ in range(0, 4)],
		'green': [(1, 0) for _ in range(0, 4)],
		'yellow': [(0, -1) for _ in range(0, 4)],
		'blue': [(-1, 0) for _ in range(0, 4)],
	}

	# create all possible game_field for each piece individually
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
		pix_empty = d_tile_name_to_pix[f'empty']

		pix_empty_mask = d_tile_name_to_pix[f'empty_mask']
		pix_empty_frame = d_tile_name_to_pix[f'empty_frame']
		pix_empty_frame_color = d_tile_name_to_pix[f'empty_frame_{player_color_1}']

		l_player_pos_abs = d_l_player_idx_pos_abs[player_color_1]

		y, x = l_player_pos_abs[0] # the first entrance of the player_color_1
		field_cell = FieldCell(
			idx_y=y, idx_x=x,
			pos_y=y*tile_size, pos_x=x*tile_size,
			name_of_field=f"entrance_{player_color_1}",
			is_starting_cell=False, is_entrance_cell=True, is_finish_cell=False,
		)
		pix_tmp = pix_empty_mask.copy()
		pix_alpha_blending_many(pix_dst=pix_tmp, l_pix_src=[pix_empty, pix_empty_frame_color], l_y=[0, 0], l_x=[0, 0])
		field_cell.d_tile_set["empty_cell"] = pix_tmp

		d_idx_pos_to_field_cell[(y, x)] = field_cell
		

		for i in range(1, 9):
			y, x = l_player_pos_abs[i] # the way of the player_color_1
			field_cell = FieldCell(
				idx_y=y, idx_x=x,
				pos_y=y*tile_size, pos_x=x*tile_size,
				name_of_field=f"road_{player_color_1}_nr_{i}",
				is_starting_cell=False, is_entrance_cell=False, is_finish_cell=False,
			)
			pix_tmp = pix_empty_mask.copy()
			pix_alpha_blending_many(pix_dst=pix_tmp, l_pix_src=[pix_empty, pix_empty_frame], l_y=[0, 0], l_x=[0, 0])
			field_cell.d_tile_set["empty_cell"] = pix_tmp

			d_idx_pos_to_field_cell[(y, x)] = field_cell
		

		y, x = l_player_pos_abs[9] # the exit field_cell of player_color_2
		field_cell = FieldCell(
			idx_y=y, idx_x=x,
			pos_y=y*tile_size, pos_x=x*tile_size,
			name_of_field=f"exit_{player_color_2}",
			is_starting_cell=False, is_entrance_cell=False, is_finish_cell=False,
		)
		pix_tmp = pix_empty_mask.copy()
		pix_alpha_blending_many(pix_dst=pix_tmp, l_pix_src=[pix_empty, pix_empty_frame], l_y=[0, 0], l_x=[0, 0])
		field_cell.d_tile_set["empty_cell"] = pix_tmp

		d_idx_pos_to_field_cell[(y, x)] = field_cell


		for i, (y, x) in enumerate(l_player_pos_abs[-4:], 1): # the last fields of the player_color_1
			field_cell = FieldCell(
				idx_y=y, idx_x=x,
				pos_y=y*tile_size, pos_x=x*tile_size,
				name_of_field=f"finish_{player_color_1}_nr_{i}",
				is_starting_cell=False, is_entrance_cell=False, is_finish_cell=True,
			)
			pix_tmp = pix_empty_mask.copy()
			pix_alpha_blending_many(pix_dst=pix_tmp, l_pix_src=[pix_empty, pix_empty_frame_color], l_y=[0, 0], l_x=[0, 0])
			field_cell.d_tile_set["empty_cell"] = pix_tmp

			d_idx_pos_to_field_cell[(y, x)] = field_cell


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
	d_player_color_to_d_piece_nr_to_piece = {}
	# d_player_color_to_l_piece_start = {}
	# d_player_color_to_l_piece_moveable = {}
	# d_player_color_to_l_piece_finish = {}
	for player_color in l_player_color:
		d_piece_nr_to_piece = {}
		l_piece = []
		for number in l_player_nr:
			piece = Piece(
				name=f'piece_{player_color}_nr_{number}',
				color=player_color,
				number=number,
			)
			
			pix_mask = d_tile_name_to_pix[f'empty_mask']
			pix_select = d_tile_name_to_pix[f'select_piece']
			
			pix_piece = d_tile_name_to_pix[f'piece_{player_color}']
			pix_number = d_tile_name_to_pix[f'nr_{number}']

			pix_tmp = pix_mask.copy()
			pix_alpha_blending_many(pix_dst=pix_tmp, l_pix_src=[pix_piece, pix_number, pix_select], l_y=[0, 0, 0], l_x=[0, 0, 0])
			piece.d_tile_set["select"] = pix_tmp
			
			pix_tmp = pix_mask.copy()
			pix_alpha_blending_many(pix_dst=pix_tmp, l_pix_src=[pix_piece, pix_number], l_y=[0, 0], l_x=[0, 0])
			piece.d_tile_set["non_select"] = pix_tmp

			d_piece_nr_to_piece[number] = piece
			l_piece.append(piece)

		d_player_color_to_d_piece_nr_to_piece[player_color] = d_piece_nr_to_piece
		# d_player_color_to_l_piece_start[player_color] = l_piece
		# d_player_color_to_l_piece_moveable[player_color] = []
		# d_player_color_to_l_piece_finish[player_color] = []

	# create the starting fields and associate the starting game_field with each piece too
	d_player_color_to_piece_home_field_cell_idx_pos = {
		'red': (0, 0),
		'green': (0, 9),
		'yellow': (9, 9),
		'blue': (9, 0),
	}

	d_player_color_to_d_starting_field_cell = {}
	for player_color in l_player_color:
		pix_empty = d_tile_name_to_pix[f'empty']
		pix_empty_frame = d_tile_name_to_pix[f'empty_frame']

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
					is_starting_cell=True, is_entrance_cell=False, is_finish_cell=False,
				)
				pix_tmp = pix_empty.copy()
				pix_alpha_blending_many(pix_dst=pix_tmp, l_pix_src=[pix_empty_frame], l_y=[0], l_x=[0])
				field_cell.d_tile_set["empty_cell"] = pix_tmp

				field_cell.d_player_color_to_field_cell_nr[player_color] = 0
				field_cell.d_player_color_to_d_next_field[player_color] = {6: starting_field_cell} # TODO: the 6 should be made generic with a const variable!

				d_idx_pos_to_field_cell[(idx_y_abs, idx_x_abs)] = field_cell

				piece = d_player_color_to_d_piece_nr_to_piece[player_color][piece_number]
				
				piece.current_field_cell = field_cell
				piece.home_field_cell = field_cell
				field_cell.current_piece = piece

				d_starting_field_cell[(idx_y_abs, idx_x_abs)] = field_cell

				piece_number += 1

		d_player_color_to_d_starting_field_cell[player_color] = d_starting_field_cell


	# now try drawing the pieces correctly

	for d_piece_nr_to_piece in d_player_color_to_d_piece_nr_to_piece.values():
		for piece in d_piece_nr_to_piece.values():
			current_field_cell = piece.current_field_cell

			y = current_field_cell.pos_y
			x = current_field_cell.pos_x

			pix_alpha_blending(pix_dst=game_field.pix, pix_src=current_field_cell.d_tile_set["empty_cell"], y=y, x=x)

			if piece.state_select:
				pix_alpha_blending(pix_dst=game_field.pix, pix_src=piece.d_tile_set["select"], y=y, x=x)
			else:
				pix_alpha_blending(pix_dst=game_field.pix, pix_src=piece.d_tile_set["non_select"], y=y, x=x)

	game_field.save_next_field_image()

	# lets try out a fixed dice roll sequence and see the result at the end!

	# create the rotation sequence of the player color
	d_player_color_to_player_color_next = {pl_col_1: pl_col_2 for pl_col_1, pl_col_2 in zip(l_player_color, l_player_color[1:]+l_player_color[:1])}
	l_dice_sequence = [
		1, 5, 6, 2,
		6, 6, 2, 6,
	]

	current_player_color = l_player_color[0]
	for dice_number in l_dice_sequence:
		# do the action first
		# l_piece_start = d_player_color_to_l_piece_start[current_player_color]
		# l_piece_moveable = d_player_color_to_l_piece_moveable[current_player_color]
		# l_piece_finish = d_player_color_to_l_piece_finish[current_player_color]

		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix[f'empty'], y=4*tile_size, x=12*tile_size)
		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix[f'piece_{current_player_color}'], y=4*tile_size, x=12*tile_size)
		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix[f'dice_nr_{dice_number}'], y=5*tile_size, x=12*tile_size)
		game_field.save_next_field_image()

		d_piece_nr_to_piece = d_player_color_to_d_piece_nr_to_piece[current_player_color]

		d_piece_nr_to_piece_start = {}
		d_piece_nr_to_piece_moveable = {}
		d_piece_nr_to_piece_finish_final = {}

		for piece_nr, piece in sorted(d_piece_nr_to_piece.items()):
			if piece.state_start:
				d_piece_nr_to_piece_start[piece_nr] = piece
			elif not piece.state_start and not piece.state_finish_final:
				d_piece_nr_to_piece_moveable[piece_nr] = piece
			elif piece.state_finish_final:
				d_piece_nr_to_piece_finish_final[piece_nr] = piece
			else:
				assert False and "Should never happen!"

		if dice_number == 6 and len(d_piece_nr_to_piece_start) > 0:
			is_move_possible = False
			for piece_nr, piece in sorted(d_piece_nr_to_piece_start.items()):
				if piece.check_if_next_move_is_possible(amount_move=dice_number):
					is_move_possible = True
					break

			if is_move_possible:
				# do the move + animation
				game_field.move_piece_from_starting_field_cell(piece=piece, amount_move=dice_number)

				pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix[f'empty'], y=4*tile_size, x=12*tile_size)
				pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix[f'dice_nr_0'], y=5*tile_size, x=12*tile_size)
				game_field.save_next_field_image()

				current_player_color = d_player_color_to_player_color_next[current_player_color]
				continue

		# TODO: add the rolled dice with the current player playing e.g. on the right side as a side info
		# TODO: make a function for the moving of a piece in the GameField class!
		# TODO: make a function for saving the new game_field.pix into images automatically, if needed
		# TODO: add data to each field_cell and piece, where each piece and dice_number and move is tracked! do this correct from the beginning
		
		# if dice_number < 6 and len(l_piece_moveable) > 0:
		# 	# move the piece which is the fartest away from the goal
		# elif dice_number == 6 and len(l_piece_start) > 0:
		# 	# move one piece out from the start, except it is not possible



		# if dice_number == 6 and len(l_piece_start) > 0: # and len(l_piece_moveable) == 0:
		# 	# need at least one moveable piece out!
		# 	piece = l_piece_start.pop(0)

		# 	if not game_field.move_piece_from_starting_field_cell(piece=piece):
		# 		l_piece_start.insert(0, piece)
		# 		# current_player_color = d_player_color_to_player_color_next[current_player_color]
				
		# 		piece = None
		# 		# continue



		# 	l_piece_moveable.append(piece)

		# 	# why not using the function move_piece_to_next_field_cell? ;-)

		# elif len(l_piece_moveable) > 0:
		# 	# move the moveable piece to the next game_field
		# 	pass
		# else:
		# 	pass

		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix[f'empty'], y=4*tile_size, x=12*tile_size)
		pix_alpha_blending(pix_dst=game_field.pix, pix_src=d_tile_name_to_pix[f'dice_nr_0'], y=5*tile_size, x=12*tile_size)
		game_field.save_next_field_image()

		current_player_color = d_player_color_to_player_color_next[current_player_color]

	# FIXME: change the priorities for each TODO if needed
	# TODO: for the ring fields (main field_cells only, not the home or the finish field_cells) for placing the piece back home, when 0 amount_move should be done!
	# TODO: make each move interactable with the updating of the playing game_field + log
	# TODO: make more functional style and oop style (do refactoring)
	# TODO: write function for the first interactions
	# TODO: make the most simplest rule for playing the game automatically
	# TODO: make possible for playing 1, 2, 3 and 4 players
	# TODO: write a log file for each game player until no more moves can be made!
	# TODO: write test cases later for testing the play game_field!

	# TODO: create the images for each tile programable with code, not hardcoded as image! (if possible)

	# img_test = Image.fromarray(pix)
	# img_test.save(os.path.join(TEMP_DIR, 'test.png'))
