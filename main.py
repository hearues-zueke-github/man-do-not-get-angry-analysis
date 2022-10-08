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

import matplotlib.pyplot as plt

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

from numpy.random import Generator, PCG64

from utils_mutliprocessing import MultiprocessingManager

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


def get_current_dt_str():
	return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H:%M:%S:%f')


# TODO: refactor game_field more cleaner
class GameField:
	def __init__(self, game_nr, h, w, l_player_color, should_save_image=True, should_save_df=True, l_seed_prefix=[]):
		self.game_nr = game_nr
		self.h = h
		self.w = w
		self.l_player_color = l_player_color
		self.should_save_image = should_save_image
		self.should_save_df = should_save_df
		self.l_seed_prefix = l_seed_prefix
		
		self.d_player_color_to_pos_idx = {player_color: pos_idx for pos_idx, player_color in enumerate(self.l_player_color, 0)}
		self.pix = np.zeros((h, w, 4), dtype=np.uint8)
		self.play_field_number = 0

		if should_save_image or should_save_df:
			self.temp_dir_path = os.path.join(os.path.join(TEMP_DIR, 'man_do_not_get_angry'), datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H:%M:%S:%f'))
			if not os.path.exists(self.temp_dir_path):
				os.makedirs(self.temp_dir_path)
		else:
			self.temp_dir_path = None

		self.l_player_color_finish = []
		self.l_pos_idx_finish = []
		self.l_piece_finish_final = []
		self.d_player_color_to_player = {}

		self.d_idx_pos_to_field_cell = {}

		self.d_player_color_to_player_color_next = {pl_col_1: pl_col_2 for pl_col_1, pl_col_2 in zip(l_player_color, l_player_color[1:]+l_player_color[:1])}
		self.d_player_color_to_player_color_next_rev = {pl_col_2: pl_col_1 for pl_col_1, pl_col_2 in self.d_player_color_to_player_color_next.items()}

		self.current_player_color = l_player_color[0]

		self.l_player_dt_str_seed = None


	def init_new_game(self):
		# create all field_cell and link them up for the next step! plus the finish field_cell for each color

		# create a incrementely tuple list beginning from the (y, x) position (AMOUNT_PIECE_FIELD_SIZE, 0)
		l_pos_inc = (
			[(0, 1) for _ in range (0, AMOUNT_PIECE_FIELD_SIZE)] +
			[(-1, 0) for _ in range (0, AMOUNT_PIECE_FIELD_SIZE)] +
			[(0, 1) for _ in range (0, 2)] +

			[(1, 0) for _ in range (0, AMOUNT_PIECE_FIELD_SIZE)] +
			[(0, 1) for _ in range (0, AMOUNT_PIECE_FIELD_SIZE)] +
			[(1, 0) for _ in range (0, 2)] +

			[(0, -1) for _ in range (0, AMOUNT_PIECE_FIELD_SIZE)] +
			[(1, 0) for _ in range (0, AMOUNT_PIECE_FIELD_SIZE)] +
			[(0, -1) for _ in range (0, 2)] +

			[(-1, 0) for _ in range (0, AMOUNT_PIECE_FIELD_SIZE)] +
			[(0, -1) for _ in range (0, AMOUNT_PIECE_FIELD_SIZE)] +
			[(-1, 0) for _ in range (0, 2)]
		)

		l_pos_abs = []
		y_abs = AMOUNT_PIECE_FIELD_SIZE + field_offset_idx_y
		x_abs = 0 + field_offset_idx_x

		for y_inc, x_inc in l_pos_inc:
			y_abs += y_inc
			x_abs += x_inc
			l_pos_abs.append((y_abs, x_abs))

		# shift the list by one element right
		l_pos_abs = l_pos_abs[-1:] + l_pos_abs[:-1]

		d_player_color_to_l_final_idx_pos_inc = {
			'red': [(0, 1) for _ in range(0, AMOUNT_PIECE_FINISH_SIZE)],
			'green': [(1, 0) for _ in range(0, AMOUNT_PIECE_FINISH_SIZE)],
			'yellow': [(0, -1) for _ in range(0, AMOUNT_PIECE_FINISH_SIZE)],
			'blue': [(-1, 0) for _ in range(0, AMOUNT_PIECE_FINISH_SIZE)],
		}

		# create all possible game_field for each piece individually
		MOVE_FIELD_CELL = AMOUNT_PIECE_FIELD_SIZE*2 + 2
		d_player_color_to_l_player_idx_pos_abs = {}
		for player_color in L_PLAYER_COLOR_ALL:
			l_final_idx_pos_inc = d_player_color_to_l_final_idx_pos_inc[player_color]

			y_abs, x_abs = l_pos_abs[-1] # get the last pos to calc the last final fields too
			l_final_pos_abs = []
			for y_inc, x_inc in l_final_idx_pos_inc:
				y_abs += y_inc
				x_abs += x_inc
				l_final_pos_abs.append((y_abs, x_abs))

			l_player_idx_pos_abs = l_pos_abs + l_final_pos_abs
			d_player_color_to_l_player_idx_pos_abs[player_color] = l_player_idx_pos_abs

			# shift the list by 10 elements left for the next piece positions
			l_pos_abs = l_pos_abs[MOVE_FIELD_CELL:] + l_pos_abs[:MOVE_FIELD_CELL]

		# and calculate a lookup table from each field_cell to the next possible field_cell, if the dice is a 6 sided one
		# e.g. from the position 2 the piece can go next to 3, 4, 5, 6, 7 and 8 because of all the possibilities

		d_player_color_to_d_lookup_table = {}
		for player_color in L_PLAYER_COLOR_ALL:
			l_player_idx_pos_abs = d_player_color_to_l_player_idx_pos_abs[player_color]

			d_lookup_table = {}
			for i in range(1, DICE_SIDES+1): # 6 sided dice
				d_lookup_table[i] = {pos_1: pos_2 for pos_1, pos_2 in zip(l_player_idx_pos_abs[:-i], l_player_idx_pos_abs[i:])}

			d_player_color_to_d_lookup_table[player_color] = d_lookup_table

		field_cell_idx = 0

		# now set for all individual field_cell the next possible field_cells
		for player_color_1, player_color_2 in zip(L_PLAYER_COLOR_ALL, L_PLAYER_COLOR_ALL[1:]+L_PLAYER_COLOR_ALL[:1]):
			pix_empty = d_tile_name_to_pix[f'empty']

			pix_empty_mask = d_tile_name_to_pix[f'empty_mask']
			pix_empty_frame = d_tile_name_to_pix[f'empty_frame']
			pix_empty_frame_color = d_tile_name_to_pix[f'empty_frame_{player_color_1}']

			l_player_idx_pos_abs = d_player_color_to_l_player_idx_pos_abs[player_color_1]

			y, x = l_player_idx_pos_abs[0] # the first entrance of the player_color_1
			field_cell = FieldCell(
				game_field=self,
				field_cell_idx=field_cell_idx,
				idx_y=y, idx_x=x,
				pos_y=y*TILE_SIZE, pos_x=x*TILE_SIZE,
				name_of_field=f"entrance_{player_color_1}",
				is_starting_cell=False, is_entrance_cell=True, is_finish_cell=False,
			)
			field_cell_idx += 1
			pix_tmp = pix_empty_mask.copy()
			pix_alpha_blending_many(pix_dst=pix_tmp, l_pix_src=[pix_empty, pix_empty_frame_color], l_y=[0, 0], l_x=[0, 0])
			field_cell.d_tile_set["empty_cell"] = pix_tmp

			self.d_idx_pos_to_field_cell[(y, x)] = field_cell
			

			# TODO: need to refactor this in the future
			for i in range(1, MOVE_FIELD_CELL-1):
				y, x = l_player_idx_pos_abs[i] # the way of the player_color_1
				field_cell = FieldCell(
					game_field=self,
					field_cell_idx=field_cell_idx,
					idx_y=y, idx_x=x,
					pos_y=y*TILE_SIZE, pos_x=x*TILE_SIZE,
					name_of_field=f"road_{player_color_1}_nr_{i}",
					is_starting_cell=False, is_entrance_cell=False, is_finish_cell=False,
				)
				field_cell_idx += 1
				pix_tmp = pix_empty_mask.copy()
				pix_alpha_blending_many(pix_dst=pix_tmp, l_pix_src=[pix_empty, pix_empty_frame], l_y=[0, 0], l_x=[0, 0])
				field_cell.d_tile_set["empty_cell"] = pix_tmp

				self.d_idx_pos_to_field_cell[(y, x)] = field_cell
			

			y, x = l_player_idx_pos_abs[MOVE_FIELD_CELL-1] # the exit field_cell of player_color_2
			field_cell = FieldCell(
				game_field=self,
				field_cell_idx=field_cell_idx,
				idx_y=y, idx_x=x,
				pos_y=y*TILE_SIZE, pos_x=x*TILE_SIZE,
				name_of_field=f"exit_{player_color_2}",
				is_starting_cell=False, is_entrance_cell=False, is_finish_cell=False,
			)
			field_cell_idx += 1
			pix_tmp = pix_empty_mask.copy()
			pix_alpha_blending_many(pix_dst=pix_tmp, l_pix_src=[pix_empty, pix_empty_frame], l_y=[0, 0], l_x=[0, 0])
			field_cell.d_tile_set["empty_cell"] = pix_tmp

			self.d_idx_pos_to_field_cell[(y, x)] = field_cell


			for i, (y, x) in enumerate(l_player_idx_pos_abs[-AMOUNT_PIECE_FINISH_SIZE:], 1): # the last fields of the player_color_1
				field_cell = FieldCell(
					game_field=self,
					field_cell_idx=field_cell_idx,
					idx_y=y, idx_x=x,
					pos_y=y*TILE_SIZE, pos_x=x*TILE_SIZE,
					name_of_field=f"finish_{player_color_1}_nr_{i}",
					is_starting_cell=False, is_entrance_cell=False, is_finish_cell=True,
				)
				field_cell_idx += 1
				pix_tmp = pix_empty_mask.copy()
				pix_alpha_blending_many(pix_dst=pix_tmp, l_pix_src=[pix_empty, pix_empty_frame_color], l_y=[0, 0], l_x=[0, 0])
				field_cell.d_tile_set["empty_cell"] = pix_tmp

				self.d_idx_pos_to_field_cell[(y, x)] = field_cell


		for player_color in L_PLAYER_COLOR_ALL:
			l_player_idx_pos_abs = d_player_color_to_l_player_idx_pos_abs[player_color]
			
			for field_cell_nr, pos_abs in enumerate(l_player_idx_pos_abs, 1):
				field_cell = self.d_idx_pos_to_field_cell[pos_abs]
				field_cell.d_player_color_to_field_cell_nr[player_color] = field_cell_nr

			d_lookup_table = d_player_color_to_d_lookup_table[player_color]

			for amount_move, d_idx_pos_1_to_idx_pos_2 in d_lookup_table.items():
				for idx_pos_1, idx_pos_2 in d_idx_pos_1_to_idx_pos_2.items():
					field_cell_1 = self.d_idx_pos_to_field_cell[idx_pos_1]
					field_cell_2 = self.d_idx_pos_to_field_cell[idx_pos_2]

					if player_color not in field_cell_1.d_player_color_to_d_next_field:
						field_cell_1.d_player_color_to_d_next_field[player_color] = {}

					field_cell_1.d_player_color_to_d_next_field[player_color][amount_move] = field_cell_2

		# create all the players with there pieces
		for player_color in self.l_player_color:
			player = Player(
				name=f'player_{player_color}',
				color=player_color,
				amount_piece=AMOUNT_PIECE_PER_PLAYER,
			)

			for number, piece in player.d_piece_nr_to_piece.items():
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

			self.d_player_color_to_player[player_color] = player

		# create the starting fields and associate the starting game_field with each piece too
		# TODO: make here the refactoring for the correct position of the starting fields
		if AMOUNT_PIECE_PER_PLAYER == 1 and AMOUNT_PIECE_FIELD_SIZE == 1:
			d_player_color_to_piece_home_field_cell_idx_pos = {
				'red': (0 + field_offset_idx_y - 1, 0 + field_offset_idx_x - 1),
				'green': (0 + field_offset_idx_y - 1, AMOUNT_PIECE_FIELD_SIZE*2 + 3 - 2 + field_offset_idx_x + 2),
				'yellow': (AMOUNT_PIECE_FIELD_SIZE*2 + 3 - 2 + field_offset_idx_y + 2, AMOUNT_PIECE_FIELD_SIZE*2 + 3 - 2 + field_offset_idx_x + 2),
				'blue': (AMOUNT_PIECE_FIELD_SIZE*2 + 3 - 2 + field_offset_idx_y + 2, 0 + field_offset_idx_x - 1),
			}
		elif AMOUNT_PIECE_PER_PLAYER == 2 and AMOUNT_PIECE_FIELD_SIZE == 2:
			d_player_color_to_piece_home_field_cell_idx_pos = {
				'red': (0 + field_offset_idx_y - 1, 0 + field_offset_idx_x - 1),
				'green': (0 + field_offset_idx_y - 1, AMOUNT_PIECE_FIELD_SIZE*2 + 3 - 2 + field_offset_idx_x + 1),
				'yellow': (AMOUNT_PIECE_FIELD_SIZE*2 + 3 - 2 + field_offset_idx_y + 2, AMOUNT_PIECE_FIELD_SIZE*2 + 3 - 2 + field_offset_idx_x + 1),
				'blue': (AMOUNT_PIECE_FIELD_SIZE*2 + 3 - 2 + field_offset_idx_y + 2, 0 + field_offset_idx_x - 1),
			}
		else:
			d_player_color_to_piece_home_field_cell_idx_pos = {
				'red': (0 + field_offset_idx_y, 0 + field_offset_idx_x),
				'green': (0 + field_offset_idx_y, AMOUNT_PIECE_FIELD_SIZE*2 + 3 - 2 + field_offset_idx_x - START_PIECE_W + 2),
				'yellow': (AMOUNT_PIECE_FIELD_SIZE*2 + 3 - 2 + field_offset_idx_y - START_PIECE_H + 2, AMOUNT_PIECE_FIELD_SIZE*2 + 3 - 2 + field_offset_idx_x - START_PIECE_W + 2),
				'blue': (AMOUNT_PIECE_FIELD_SIZE*2 + 3 - 2 + field_offset_idx_y  - START_PIECE_H + 2, 0 + field_offset_idx_x),
			}

		# d_player_color_to_d_starting_field_cell = {}
		for player_color in self.l_player_color:
			player = self.d_player_color_to_player[player_color]

			pix_empty = d_tile_name_to_pix[f'empty']
			pix_empty_frame = d_tile_name_to_pix[f'empty_frame']

			l_player_idx_pos_abs = d_player_color_to_l_player_idx_pos_abs[player_color]
			starting_player_pos_abs = l_player_idx_pos_abs[0]
			starting_field_cell = self.d_idx_pos_to_field_cell[starting_player_pos_abs]

			starting_piece_idx_pos = d_player_color_to_piece_home_field_cell_idx_pos[player_color]
			idx_y, idx_x = starting_piece_idx_pos

			# TODO: refactor this part more dynamically
			l_pos_abs = [(idx_y+y_inc, idx_x+x_inc) for y_inc in range(0, START_PIECE_H) for x_inc in range(0, START_PIECE_W)]
			for piece_number, (idx_y_abs, idx_x_abs) in enumerate(l_pos_abs, 1):
				if piece_number not in player.d_piece_nr_to_piece:
					continue

				piece = player.d_piece_nr_to_piece[piece_number]

				field_cell = FieldCell(
					game_field=self,
					field_cell_idx=field_cell_idx,
					idx_y=idx_y_abs, idx_x=idx_x_abs,
					pos_y=idx_y_abs*TILE_SIZE, pos_x=idx_x_abs*TILE_SIZE,
					name_of_field=f"starting_{player_color_1}_nr_{piece_number}",
					is_starting_cell=True, is_entrance_cell=False, is_finish_cell=False,
				)
				field_cell_idx += 1
				pix_tmp = pix_empty.copy()
				pix_alpha_blending_many(pix_dst=pix_tmp, l_pix_src=[pix_empty_frame], l_y=[0], l_x=[0])
				field_cell.d_tile_set["empty_cell"] = pix_tmp

				field_cell.d_player_color_to_field_cell_nr[player_color] = 0
				field_cell.d_player_color_to_d_next_field[player_color] = {DICE_SIDES: starting_field_cell} # TODO: the 6 should be made generic with a const variable!

				self.d_idx_pos_to_field_cell[(idx_y_abs, idx_x_abs)] = field_cell
				
				piece.current_field_cell = field_cell
				piece.home_field_cell = field_cell
				field_cell.current_piece = piece


		# TODO: refactor this part of the code!
		# draw the entire field with the empty tile
		for y in range(0, self.h, TILE_SIZE):
			for x in range(0, self.w, TILE_SIZE):
				pix_alpha_blending(pix_dst=self.pix, pix_src=d_tile_name_to_pix['empty'], y=y, x=x)

		# tile_name, y, x
		l_tile_name_place = [
			('side_bar_info_color_dice', SIDE_IDX_Y*TILE_SIZE, SIDE_IDX_X*TILE_SIZE),
			('dice_nr_0', SIDE_DICE_IDX_Y*TILE_SIZE, SIDE_DICE_IDX_X*TILE_SIZE),

			('arrow_right', ARROW_RIGHT_IDX_Y*TILE_SIZE, ARROW_RIGHT_IDX_X*TILE_SIZE),
			('arrow_down', ARROW_DOWN_IDX_Y*TILE_SIZE, ARROW_DOWN_IDX_X*TILE_SIZE),
			('arrow_left', ARROW_LEFT_IDX_Y*TILE_SIZE, ARROW_LEFT_IDX_X*TILE_SIZE),
			('arrow_up', ARROW_UP_IDX_Y*TILE_SIZE, ARROW_UP_IDX_X*TILE_SIZE),
		]
		for tile_name, y, x in l_tile_name_place:
			pix_alpha_blending(pix_dst=self.pix, pix_src=d_tile_name_to_pix[tile_name], y=y, x=x)

		# draw all the field_cells
		for field_cell in self.d_idx_pos_to_field_cell.values():
			pix_alpha_blending(pix_dst=self.pix, pix_src=field_cell.d_tile_set["empty_cell"], y=field_cell.pos_y, x=field_cell.pos_x)

		self.save_next_field_image()

		# draw all the pieces
		for player in self.d_player_color_to_player.values():
			for piece in player.d_piece_nr_to_piece.values():
				current_field_cell = piece.current_field_cell

				y = current_field_cell.pos_y
				x = current_field_cell.pos_x

				if piece.state_select:
					pix_alpha_blending(pix_dst=self.pix, pix_src=piece.d_tile_set["select"], y=y, x=x)
				else:
					pix_alpha_blending(pix_dst=self.pix, pix_src=piece.d_tile_set["non_select"], y=y, x=x)

		self.save_next_field_image()
		# Image.fromarray(self.pix).show()

		# # lets try out a fixed dice roll sequence and see the result at the end!
		# # example for the random l_dice_sequence: np.tile(np.arange(1, 7), 2)[np.random.permutation(np.arange(0, 12))]
		# d_player_color_to_d_l_dice_sequence_idx = {
		# 	'red': {'l_dice_sequence': [6, 4, 5, 1, 2, 1, 3, 6, 2, 3, 5, 4], 'idx': 0},
		# 	'green': {'l_dice_sequence': [4, 4, 2, 1, 6, 5, 2, 5, 1, 3, 3, 6], 'idx': 0},
		# 	'yellow': {'l_dice_sequence': [6, 4, 1, 5, 4, 2, 5, 3, 1, 6, 3, 2], 'idx': 0},
		# 	'blue': {'l_dice_sequence': [5, 4, 6, 6, 5, 1, 2, 3, 3, 4, 1, 2], 'idx': 0},
		# }

		# for player_color in l_player_color:
		# 	player = self.d_player_color_to_player[player_color]

		l_player_dt_str_seed = []
		self.l_player_dt_str_seed = l_player_dt_str_seed

		for player in self.d_player_color_to_player.values():
			dt_str = get_current_dt_str()
			seed = self.l_seed_prefix + list(dt_str.encode('utf-8')) + list(bytes(np.uint64([self.game_nr]).view(np.uint8)))
			
			l_player_dt_str_seed.append((player, dt_str, seed))

			player.rnd = Generator(PCG64(seed))


	def save_next_field_image(self):
		if self.should_save_image:
			Image.fromarray(self.pix).save(os.path.join(self.temp_dir_path, f'field_nr_{self.play_field_number:05}.png'))
			print(f"Saving next frame: self.play_field_number: {self.play_field_number}")
			self.play_field_number += 1


	def get_df_data(self):
		d_name_to_df = {}

		l_column = [
			'game_nr', 'player_color', 'player_color_idx', 'pos_idx', 'place',
			'amount_dice_roll', 'amount_move',
		]
		d_data = {column: [] for column in l_column}
		d_player_color_to_place = {player_color: place for place, player_color in enumerate(self.l_player_color_finish, 1)}

		for player_color, player in self.d_player_color_to_player.items():
			d_data['game_nr'].append(self.game_nr)
			
			player_color_idx = D_PLAYER_COLOR_TO_PLAYER_COLOR_IDX[player_color]

			d_data['player_color'].append(player_color)
			d_data['player_color_idx'].append(player_color_idx)

			pos_idx = self.d_player_color_to_pos_idx[player_color]
			d_data['pos_idx'].append(pos_idx)
			
			place = d_player_color_to_place[player_color]
			d_data['place'].append(place)

			sum_amount_move = sum([len(piece.l_amount_move) for piece in player.d_piece_nr_to_piece.values()])
			amount_dice_roll = len(player.l_dice_number)

			assert amount_dice_roll >= sum_amount_move

			d_data['amount_dice_roll'].append(amount_dice_roll)
			d_data['amount_move'].append(sum_amount_move)

		df_player_stats = pd.DataFrame(data=d_data, columns=l_column, dtype=object)

		df_player_stats_sort = df_player_stats.sort_values(by=['player_color_idx', 'pos_idx'])
		d_name_to_df['df_player_stats'] = df_player_stats_sort


		l_column = [
			'player_color', 'player_color_idx',
			'piece', 'piece_nr',
			'amount_move', 'amount_beat_by_any', 'amount_beat_any',
			'amount_beat_by_red', 'amount_beat_by_green', 'amount_beat_by_yellow', 'amount_beat_by_blue',
			'amount_beat_red', 'amount_beat_green', 'amount_beat_yellow', 'amount_beat_blue',
		]
		d_data = {column: [] for column in l_column}

		for player_color, player in self.d_player_color_to_player.items():
			player_color_idx = D_PLAYER_COLOR_TO_PLAYER_COLOR_IDX[player_color]

			for piece_nr, piece in player.d_piece_nr_to_piece.items():
				d_data['player_color'].append(player_color)
				d_data['player_color_idx'].append(player_color_idx)

				d_data['piece'].append(piece)
				d_data['piece_nr'].append(piece_nr)

				d_amount_move = dict(piece.l_amount_move)
				d_removed_by_piece = dict(piece.l_removed_by_piece)
				d_removed_the_piece = dict(piece.l_removed_the_piece)
				
				amount_move = len(d_amount_move)
				amount_beat_by_any = len(d_removed_by_piece)
				amount_beat_any = len(d_removed_the_piece)

				d_data['amount_move'].append(amount_move)
				d_data['amount_beat_by_any'].append(amount_beat_by_any)
				d_data['amount_beat_any'].append(amount_beat_any)

				d_stat_beat_by = {color: 0 for color in L_PLAYER_COLOR_ALL}
				d_stat_beat = {color: 0 for color in L_PLAYER_COLOR_ALL}

				for other_piece in d_removed_by_piece.values():
					d_stat_beat_by[other_piece.color] += 1

				for other_piece in d_removed_the_piece.values():
					d_stat_beat[other_piece.color] += 1

				for color, amount_beat_by in d_stat_beat_by.items():
					d_data[f'amount_beat_by_{color}'].append(amount_beat_by)

				for color, amount_beat in d_stat_beat.items():
					d_data[f'amount_beat_{color}'].append(amount_beat)

		df_piece_stats = pd.DataFrame(data=d_data, columns=l_column, dtype=object)

		df_piece_stats_sort = df_piece_stats.sort_values(by=['player_color_idx', 'piece_nr']).drop(labels=['piece'], axis=1)
		d_name_to_df['df_piece_stats'] = df_piece_stats_sort


		d_action_type_to_action_type_idx = {
			'amount_move,is_in_finish': 1,
			'player_color,piece_nr': 2,
		}

		l_column = [
			'step_turn',
			'action_type', 'action_value', 'action_idx',
			'player_color', 'player_color_idx',
			'piece', 'piece_nr',
			'field_cell_from', 'field_cell_from_idx', 'field_cell_from_idx_y', 'field_cell_from_idx_x',
			'field_cell_to', 'field_cell_to_idx', 'field_cell_to_idx_y', 'field_cell_to_idx_x',
			'is_in_finish',
		]
		d_data = {column: [] for column in l_column}

		for player_color, player in self.d_player_color_to_player.items():
			player_color_idx = D_PLAYER_COLOR_TO_PLAYER_COLOR_IDX[player_color]

			for piece_nr, piece in player.d_piece_nr_to_piece.items():
				d_amount_move = dict(piece.l_amount_move)

				d_going_from_field_cell = dict(piece.l_going_from_field_cell)
				d_going_to_field_cell = dict(piece.l_going_to_field_cell)
				d_removed_by_piece = dict(piece.l_removed_by_piece)

				assert set(d_going_from_field_cell.keys()) == set(d_going_to_field_cell.keys())

				d_step_turn_to_d_field_cell = {
					step_turn: {
						'going_from': d_going_from_field_cell[step_turn],
						'going_to': d_going_to_field_cell[step_turn],
					}
					for step_turn in d_going_to_field_cell.keys()
				}

				for step_turn, d_field_cell in d_step_turn_to_d_field_cell.items():
					field_cell_from = d_field_cell['going_from']
					field_cell_to = d_field_cell['going_to']

					d_data['step_turn'].append(step_turn)

					d_data['player_color'].append(player_color)
					d_data['player_color_idx'].append(player_color_idx)

					d_data['piece'].append(piece)
					d_data['piece_nr'].append(piece_nr)

					is_in_finish = int(False if piece.step_turn_in_finish is None else piece.step_turn_in_finish == step_turn)
					if step_turn in d_amount_move:
						d_data['action_type'].append(f'amount_move,is_in_finish')
						d_data['action_value'].append(f'{d_amount_move[step_turn]},{is_in_finish}')
						d_data['action_idx'].append(d_action_type_to_action_type_idx[f'amount_move,is_in_finish'])
					else:
						piece_attacker = d_removed_by_piece[step_turn]
						
						d_data['action_type'].append(f'player_color,piece_nr')
						d_data['action_value'].append(f'{piece_attacker.color},{piece_attacker.number}')
						d_data['action_idx'].append(d_action_type_to_action_type_idx[f'player_color,piece_nr'])

					d_data['field_cell_from'].append(field_cell_from)
					d_data['field_cell_from_idx'].append(field_cell_from.field_cell_idx)
					d_data['field_cell_from_idx_y'].append(field_cell_from.idx_y)
					d_data['field_cell_from_idx_x'].append(field_cell_from.idx_x)

					d_data['field_cell_to'].append(field_cell_to)
					d_data['field_cell_to_idx'].append(field_cell_to.field_cell_idx)
					d_data['field_cell_to_idx_y'].append(field_cell_to.idx_y)
					d_data['field_cell_to_idx_x'].append(field_cell_to.idx_x)

					d_data['is_in_finish'].append(int(is_in_finish))

		df_piece_action = pd.DataFrame(data=d_data, columns=l_column, dtype=object)

		df_piece_action_sort = df_piece_action.sort_values(by=['step_turn', 'action_idx']).drop(labels=['piece', 'field_cell_from', 'field_cell_to'], axis=1)
		d_name_to_df['df_piece_action'] = df_piece_action_sort


		return d_name_to_df


	def save_df_data(self, d_name_to_df):
		if not self.should_save_df:
			return None

		for name, df in d_name_to_df.items():
			df.to_csv(os.path.join(self.temp_dir_path, f'{name}.csv'), sep=';', index=False)


	def move_piece(self, piece: 'Piece', step_turn: int, amount_move: int):
		current_field_cell = piece.current_field_cell
		other_piece, is_moved = current_field_cell.move_piece_to_next_field_cell(step_turn=step_turn, amount_move=amount_move)
		assert is_moved
		next_field_cell = piece.current_field_cell

		if next_field_cell.is_finish_cell:
			piece.state_finish = True
			if piece.step_turn_in_finish is None:
				piece.step_turn_in_finish = step_turn

		assert (other_piece is None) or (other_piece is not None and other_piece.color != piece.color)

		pos_y_1 = current_field_cell.pos_y
		pos_x_1 = current_field_cell.pos_x
		pos_y_2 = next_field_cell.pos_y
		pos_x_2 = next_field_cell.pos_x

		if self.should_save_image:
			piece.state_select = True
			pix_alpha_blending(pix_dst=self.pix, pix_src=current_field_cell.d_tile_set["empty_cell"], y=pos_y_1, x=pos_x_1)
			if piece.state_select:
				pix_alpha_blending(pix_dst=self.pix, pix_src=piece.d_tile_set["select"], y=pos_y_1, x=pos_x_1)
			else:
				pix_alpha_blending(pix_dst=self.pix, pix_src=piece.d_tile_set["non_select"], y=pos_y_1, x=pos_x_1)
			# self.save_next_field_image()
			
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
			# self.save_next_field_image()

			piece.state_select = False
			pix_alpha_blending(pix_dst=self.pix, pix_src=next_field_cell.d_tile_set["empty_cell"], y=pos_y_2, x=pos_x_2)
			if piece.state_select:
				pix_alpha_blending(pix_dst=self.pix, pix_src=piece.d_tile_set["select"], y=pos_y_2, x=pos_x_2)
			else:
				pix_alpha_blending(pix_dst=self.pix, pix_src=piece.d_tile_set["non_select"], y=pos_y_2, x=pos_x_2)
			self.save_next_field_image()

		return True


	def check_if_last_player_finished_otherwise_next_player(self, player, step_turn):
		for piece in player.d_piece_nr_to_piece.values():
			if piece.state_start:
				continue

			# check if all other next possible moves are not possible. if not then this piece is in the finish_final state
			d_player_color_to_d_next_field = piece.current_field_cell.d_player_color_to_d_next_field

			if len(d_player_color_to_d_next_field) == 0:
				piece.state_finish = False
				piece.state_finish_final = True
				self.l_piece_finish_final.append(piece)
				continue

			d_next_field = d_player_color_to_d_next_field[piece.color]

			is_move_possible = False
			for next_field_cell in d_next_field.values():
				if (next_field_cell.current_piece is None) or (next_field_cell.current_piece.color != piece.color):
					is_move_possible = True
					break

			if not is_move_possible:
				piece.state_finish = False
				piece.state_finish_final = True

		count_state_finish_final = 0
		for piece in player.d_piece_nr_to_piece.values():
			count_state_finish_final += piece.state_finish_final

		if count_state_finish_final == AMOUNT_PIECE_PER_PLAYER:
			# remove the self.current_player_color from the queue!
			self.l_player_color_finish.append(self.current_player_color)
			self.l_pos_idx_finish.append(self.d_player_color_to_pos_idx[self.current_player_color])

			if len(self.d_player_color_to_player_color_next) <= 1:
				return True

			prev_player_color = self.d_player_color_to_player_color_next_rev[self.current_player_color]
			next_player_color = self.d_player_color_to_player_color_next[self.current_player_color]

			del self.d_player_color_to_player_color_next[self.current_player_color]
			del self.d_player_color_to_player_color_next_rev[self.current_player_color]

			self.d_player_color_to_player_color_next[prev_player_color] = next_player_color
			self.d_player_color_to_player_color_next_rev[next_player_color] = prev_player_color

			self.current_player_color = next_player_color
		else:
			self.current_player_color = self.d_player_color_to_player_color_next[self.current_player_color]#

		return False


	def play_the_game(self, max_step_turn=100):
		for step_turn in range(1, max_step_turn+1):
			if step_turn % 1000 == 0:
				print(f"step_turn: {step_turn:5}, current_player_color: {self.current_player_color}")

			player = self.d_player_color_to_player[self.current_player_color]
			dice_number = player.get_next_dice_number(step_turn=step_turn)

			if self.should_save_image:
				pix_alpha_blending(pix_dst=self.pix, pix_src=d_tile_name_to_pix[f'empty'], y=SIDE_PIECE_IDX_Y*TILE_SIZE, x=SIDE_PIECE_IDX_X*TILE_SIZE)
				pix_alpha_blending(pix_dst=self.pix, pix_src=d_tile_name_to_pix[f'piece_{self.current_player_color}'], y=SIDE_PIECE_IDX_Y*TILE_SIZE, x=SIDE_PIECE_IDX_X*TILE_SIZE)
				pix_alpha_blending(pix_dst=self.pix, pix_src=d_tile_name_to_pix[f'dice_nr_{dice_number}'], y=SIDE_DICE_IDX_Y*TILE_SIZE, x=SIDE_DICE_IDX_X*TILE_SIZE)
				self.save_next_field_image()

			d_piece_nr_to_piece_start = {}
			d_piece_nr_to_piece_moveable = {}
			d_piece_nr_to_piece_finish_final = {}

			for piece_nr, piece in sorted(player.d_piece_nr_to_piece.items()):
				if piece.state_start:
					d_piece_nr_to_piece_start[piece_nr] = piece
				elif not piece.state_start and not piece.state_finish_final:
					d_piece_nr_to_piece_moveable[piece_nr] = piece
				elif piece.state_finish_final:
					d_piece_nr_to_piece_finish_final[piece_nr] = piece
				else:
					assert False and "Should never happen!"

			if dice_number == DICE_SIDES and len(d_piece_nr_to_piece_start) > 0:
				is_move_possible = False
				for piece_nr, piece in sorted(d_piece_nr_to_piece_start.items()):
					if piece.check_if_next_move_is_possible(amount_move=dice_number):
						is_move_possible = True
						break

				if is_move_possible:
					# do the move + animation
					if self.should_save_image:
						pix_alpha_blending(pix_dst=self.pix, pix_src=d_tile_name_to_pix[f'empty'], y=SIDE_PIECE_IDX_Y*TILE_SIZE, x=SIDE_PIECE_IDX_X*TILE_SIZE)
						pix_alpha_blending(pix_dst=self.pix, pix_src=piece.d_tile_set['non_select'], y=SIDE_PIECE_IDX_Y*TILE_SIZE, x=SIDE_PIECE_IDX_X*TILE_SIZE)
					
					self.move_piece(piece=piece, step_turn=step_turn, amount_move=dice_number)

					if self.should_save_image:
						pix_alpha_blending(pix_dst=self.pix, pix_src=d_tile_name_to_pix[f'empty'], y=SIDE_PIECE_IDX_Y*TILE_SIZE, x=SIDE_PIECE_IDX_X*TILE_SIZE)
						pix_alpha_blending(pix_dst=self.pix, pix_src=d_tile_name_to_pix[f'dice_nr_0'], y=SIDE_DICE_IDX_Y*TILE_SIZE, x=SIDE_DICE_IDX_X*TILE_SIZE)
						self.save_next_field_image()

					if self.check_if_last_player_finished_otherwise_next_player(player=player, step_turn=step_turn):
						break
					continue

			if len(d_piece_nr_to_piece_moveable) > 0:
				# find the least furthest one and check, if it can be moved or not!
				d_field_cell_nr_to_piece_nr = {
					piece.current_field_cell.d_player_color_to_field_cell_nr[self.current_player_color]: piece_nr
					for piece_nr, piece in d_piece_nr_to_piece_moveable.items()
				}
				
				arr_field_cell_nr_piece_nr = np.array(list(d_field_cell_nr_to_piece_nr.items()), dtype=object)
				arr_idx_permutate = player.rnd.permutation(np.arange(0, arr_field_cell_nr_piece_nr.shape[0]))
				arr_field_cell_nr_piece_nr_shuffle = arr_field_cell_nr_piece_nr[arr_idx_permutate]
				# l_field_cell_nr_piece_nr_sort = sorted(d_field_cell_nr_to_piece_nr.items())

				is_move_possible = False
				# for _, piece_nr in l_field_cell_nr_piece_nr_sort:
				for _, piece_nr in arr_field_cell_nr_piece_nr_shuffle:
					piece = d_piece_nr_to_piece_moveable[piece_nr]
					if piece.check_if_next_move_is_possible(amount_move=dice_number):
						is_move_possible = True
						break

				if is_move_possible:
					# do the move + animation
					if self.should_save_image:
						pix_alpha_blending(pix_dst=self.pix, pix_src=d_tile_name_to_pix[f'empty'], y=SIDE_PIECE_IDX_Y*TILE_SIZE, x=SIDE_PIECE_IDX_X*TILE_SIZE)
						pix_alpha_blending(pix_dst=self.pix, pix_src=piece.d_tile_set['non_select'], y=SIDE_PIECE_IDX_Y*TILE_SIZE, x=SIDE_PIECE_IDX_X*TILE_SIZE)
					
					self.move_piece(piece=piece, step_turn=step_turn, amount_move=dice_number)

					if self.should_save_image:
						pix_alpha_blending(pix_dst=self.pix, pix_src=d_tile_name_to_pix[f'empty'], y=SIDE_PIECE_IDX_Y*TILE_SIZE, x=SIDE_PIECE_IDX_X*TILE_SIZE)
						pix_alpha_blending(pix_dst=self.pix, pix_src=d_tile_name_to_pix[f'dice_nr_0'], y=SIDE_DICE_IDX_Y*TILE_SIZE, x=SIDE_DICE_IDX_X*TILE_SIZE)
						self.save_next_field_image()

					if self.check_if_last_player_finished_otherwise_next_player(player=player, step_turn=step_turn):
						break
					continue

			if self.should_save_image:
				pix_alpha_blending(pix_dst=self.pix, pix_src=d_tile_name_to_pix[f'empty'], y=SIDE_PIECE_IDX_Y*TILE_SIZE, x=SIDE_PIECE_IDX_X*TILE_SIZE)
				pix_alpha_blending(pix_dst=self.pix, pix_src=d_tile_name_to_pix[f'dice_nr_0'], y=SIDE_DICE_IDX_Y*TILE_SIZE, x=SIDE_DICE_IDX_X*TILE_SIZE)
				self.save_next_field_image()

			self.current_player_color = self.d_player_color_to_player_color_next[self.current_player_color]

		self.step_turn = step_turn


class FieldCell:
	def __init__(self, game_field, field_cell_idx, idx_y, idx_x, pos_y, pos_x, name_of_field, is_starting_cell, is_entrance_cell, is_finish_cell):
		self.game_field = game_field
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
		# add the step_turn info with the piece
		self.l_comming_piece = []
		self.l_going_piece = []

		self.field_cell_idx = field_cell_idx


	def log_comming_piece(self, step_turn, piece):
		self.l_comming_piece.append((step_turn, piece))


	def log_going_piece(self, step_turn, piece):
		self.l_going_piece.append((step_turn, piece))


	# the other field_cell must be empty, otherwise there will be a contradiction!
	def move_piece_to_next_field_cell(self, step_turn: int, amount_move: int) -> bool:
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
			
			other_piece.log_removed_by_piece(step_turn=step_turn, piece=current_piece)
			current_piece.log_removed_the_piece(step_turn=step_turn, piece=other_piece)

			other_piece.current_field_cell.log_going_piece(step_turn=step_turn, piece=other_piece)
			other_piece.log_going_from_field_cell(step_turn=step_turn, field_cell=other_piece.current_field_cell)

			other_piece.current_field_cell = other_piece.home_field_cell
			other_piece.current_field_cell.current_piece = other_piece
			other_piece.current_field_cell.log_comming_piece(step_turn=step_turn, piece=other_piece)
			other_piece.log_going_to_field_cell(step_turn=step_turn, field_cell=other_piece.current_field_cell)

			other_piece.state_start = True
		else:
			if current_piece.state_start:
				current_piece.state_start = False

		self.log_going_piece(step_turn=step_turn, piece=current_piece)
		current_piece.log_going_from_field_cell(step_turn=step_turn, field_cell=self)

		current_piece.current_field_cell = next_field_cell # update the current_field_cell of the found current_piece
		self.current_piece = None

		next_field_cell.current_piece = current_piece
		next_field_cell.log_comming_piece(step_turn=step_turn, piece=current_piece)
		current_piece.log_going_to_field_cell(step_turn=step_turn, field_cell=next_field_cell)

		current_piece.log_amount_move(step_turn=step_turn, amount_move=amount_move)

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
		player: 'Player',
		name: str,
		color: str,
		number: int,
	):
		self.player = player
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
		# add the step_turn info with the field_cell
		self.l_going_to_field_cell = []
		self.l_going_from_field_cell = []
		self.l_amount_move = []
		self.l_removed_by_piece = []
		self.l_removed_the_piece = []
		self.step_turn_in_finish = None


	def log_going_to_field_cell(self, step_turn, field_cell):
		self.l_going_to_field_cell.append((step_turn, field_cell))


	def log_going_from_field_cell(self, step_turn, field_cell):
		self.l_going_from_field_cell.append((step_turn, field_cell))


	def log_amount_move(self, step_turn, amount_move):
		self.l_amount_move.append((step_turn, amount_move))


	def log_removed_by_piece(self, step_turn, piece):
		self.l_removed_by_piece.append((step_turn, piece))


	def log_removed_the_piece(self, step_turn, piece):
		self.l_removed_the_piece.append((step_turn, piece))


	def check_if_next_move_is_possible(self, amount_move):
		color = self.color
		d_player_color_to_d_next_field = self.current_field_cell.d_player_color_to_d_next_field

		if color not in d_player_color_to_d_next_field:
			return False

		d_next_field = d_player_color_to_d_next_field[color]

		if amount_move not in d_next_field:
			return False

		next_field_cell = d_next_field[amount_move]

		if next_field_cell.current_piece is not None and next_field_cell.current_piece.color == self.color:
			return False

		return True


class Player:
	def __init__(
		self,
		name,
		color,
		amount_piece,
		rnd=None,
	):
		self.name = name
		self.color = color
		self.amount_piece = amount_piece
		self.d_piece_nr_to_piece = {
			number: Piece(
				player=self,
				name=f'piece_{color}_nr_{number}',
				color=color,
				number=number,
			)
			for number in range(1, amount_piece+1)
		}
		self.l_dice_sequence = [i for i in range(1, DICE_SIDES+1)] # default l_dice_sequence
		self.idx_dice = 0
		self.amount_dice_used = 0
		self.l_dice_number = []

		self.rnd = rnd


	def get_next_dice_number(self, step_turn):
		if self.rnd is None:
			dice_number = self.l_dice_sequence[self.idx_dice]
			self.idx_dice = (self.idx_dice + 1) % len(self.l_dice_sequence)
		else:
			dice_number = self.rnd.integers(1, DICE_SIDES+1)

		self.l_dice_number.append((step_turn, dice_number)) # with the step_turn saved
		self.amount_dice_used += 1

		return dice_number


def load_tiles():
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
	pix_number_5 = np.array(Image.open(os.path.join(dir_img_path, 'number_5_16x16.png')))
	pix_number_6 = np.array(Image.open(os.path.join(dir_img_path, 'number_6_16x16.png')))
	pix_number_7 = np.array(Image.open(os.path.join(dir_img_path, 'number_7_16x16.png')))
	pix_number_8 = np.array(Image.open(os.path.join(dir_img_path, 'number_8_16x16.png')))
	pix_number_9 = np.array(Image.open(os.path.join(dir_img_path, 'number_9_16x16.png')))
	
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
	
	pix_side_bar_info_color_dice = np.array(Image.open(os.path.join(dir_img_path, 'side_bar_info_color_dice_80x48.png')))

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
		'nr_5': pix_number_5,
		'nr_6': pix_number_6,
		'nr_7': pix_number_7,
		'nr_8': pix_number_8,
		'nr_9': pix_number_9,
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
		'side_bar_info_color_dice': pix_side_bar_info_color_dice,
	}

	return d_tile_name_to_pix


if __name__ == '__main__':
	dir_img_path = os.path.join(CURRENT_WORKING_DIR, 'img')
	assert os.path.exists(dir_img_path)

	argv = sys.argv
	assert len(argv) == 4

	AMOUNT_PIECE_PER_PLAYER = int(argv[1])
	AMOUNT_PIECE_FINISH_SIZE = int(argv[2])
	AMOUNT_PIECE_FIELD_SIZE = int(argv[3])

	assert AMOUNT_PIECE_PER_PLAYER <= AMOUNT_PIECE_FINISH_SIZE
	assert AMOUNT_PIECE_FINISH_SIZE <= AMOUNT_PIECE_FIELD_SIZE
	
	# L_PLAYER_COLOR_IDX = [int(v_str) for v_str in argv[5].split(',')]
	# assert len(L_PLAYER_COLOR_IDX) == amount_player

	# u, c = np.unique(L_PLAYER_COLOR_IDX, return_counts=True)
	# assert u.shape[0] == amount_player
	# assert np.all(c == 1)

	L_PLAYER_COLOR_ALL = ['red', 'green', 'yellow', 'blue']
	D_PLAYER_COLOR_TO_PLAYER_COLOR_IDX = {player_color: player_color_idx for player_color_idx, player_color in enumerate(L_PLAYER_COLOR_ALL, 1)}
	D_PLAYER_COLOR_IDX_TO_PLAYER_COLOR = {player_color_idx: player_color for player_color_idx, player_color in enumerate(L_PLAYER_COLOR_ALL, 1)}


	l_l_player_color_idx = [
		[1],

		[1, 2],
		[2, 1],
		[1, 3],
		[3, 1],

		[1, 2, 3],
		[1, 3, 2],
		[2, 1, 3],
		[2, 3, 1],
		[3, 2, 1],
		[3, 1, 2],

		[1, 4, 2, 3],
		[1, 4, 3, 2],
		[1, 2, 4, 3],
		[1, 2, 3, 4],
		[1, 3, 2, 4],
		[1, 3, 4, 2],
	]

	l_df_stats_avrg = []
	for l_player_color_idx in l_l_player_color_idx:
		amount_player = len(l_player_color_idx)

		l_player_color = [D_PLAYER_COLOR_IDX_TO_PLAYER_COLOR[player_color_idx] for player_color_idx in l_player_color_idx]

		start_piece_w = 0
		start_piece_h = 0
		for _ in range(0, 100):
			start_piece_w += 1
			if start_piece_w * start_piece_h >= AMOUNT_PIECE_PER_PLAYER:
				break

			start_piece_h += 1
			if start_piece_w * start_piece_h >= AMOUNT_PIECE_PER_PLAYER:
				break

		START_PIECE_W = start_piece_w
		START_PIECE_H = start_piece_h

		DICE_SIDES = 6

		TILE_SIZE = 16
		tiles_amount_field = AMOUNT_PIECE_FIELD_SIZE*2 + 3
		
		field_offset_idx_y = 2
		field_offset_idx_x = 2
		
		TILES_AMOUNT_H = tiles_amount_field + field_offset_idx_y*2
		TILES_AMOUNT_W = tiles_amount_field + field_offset_idx_x*2 + 3


		# TODO: calc the needed idx_x for the side constants
		SIDE_IDX_Y = 2
		SIDE_IDX_X = TILES_AMOUNT_H
		SIDE_PIECE_IDX_Y = 3
		SIDE_PIECE_IDX_X = TILES_AMOUNT_H + 1
		SIDE_DICE_IDX_Y = 5
		SIDE_DICE_IDX_X = TILES_AMOUNT_H + 1

		ARROW_RIGHT_IDX_Y = AMOUNT_PIECE_FIELD_SIZE - 1 + field_offset_idx_y
		ARROW_RIGHT_IDX_X = 0 + field_offset_idx_x
		ARROW_DOWN_IDX_Y = 0 + field_offset_idx_y
		ARROW_DOWN_IDX_X = AMOUNT_PIECE_FIELD_SIZE + 3 + field_offset_idx_x
		ARROW_LEFT_IDX_Y = AMOUNT_PIECE_FIELD_SIZE + 3 + field_offset_idx_y
		ARROW_LEFT_IDX_X = AMOUNT_PIECE_FIELD_SIZE*2 + 3 - 2 + field_offset_idx_x
		ARROW_UP_IDX_Y = AMOUNT_PIECE_FIELD_SIZE*2 + 3 - 2 + field_offset_idx_y
		ARROW_UP_IDX_X = AMOUNT_PIECE_FIELD_SIZE - 1 + field_offset_idx_x

		MAX_STEP_TURN = 100000

		SHOULD_SAVE_IMAGE = False
		SHOULD_SAVE_DF = False

		if AMOUNT_PIECE_FIELD_SIZE == 1:
			ARROW_RIGHT_IDX_X -= 1
			ARROW_DOWN_IDX_Y -= 1
			ARROW_LEFT_IDX_X += 1
			ARROW_UP_IDX_Y += 1

		PIX_H = TILES_AMOUNT_H * TILE_SIZE
		PIX_W = TILES_AMOUNT_W * TILE_SIZE

		d_tile_name_to_pix = load_tiles()

		CPU_COUNT = mp.cpu_count()

		MAX_GAME_NR = 500
		CHUNK_SIZE = MAX_GAME_NR // ((CPU_COUNT - 1) * 4)

		mult_proc_mng = MultiprocessingManager(cpu_count=CPU_COUNT)

		# # only for testing the responsivness!
		# mult_proc_mng.test_worker_threads_response()

		def play_some_games(l_game_nr):
			# l_game_field = []
			l_d_name_to_df = []
			l_step_turn = []

			for game_nr in l_game_nr:
				# print(f'game_nr: {game_nr:5}/{MAX_GAME_NR:5}')
				game_field = GameField(
					game_nr=game_nr,
					h=PIX_H,
					w=PIX_W,
					l_player_color=l_player_color,
					should_save_image=SHOULD_SAVE_IMAGE,
					should_save_df=SHOULD_SAVE_DF,
					l_seed_prefix=[],
				)

				game_field.init_new_game()
				
				game_field.play_the_game(max_step_turn=MAX_STEP_TURN)
				d_name_to_df = game_field.get_df_data()
				game_field.save_df_data(d_name_to_df=d_name_to_df)

				# l_game_field.append(game_field)
				l_d_name_to_df.append(d_name_to_df)
				l_step_turn.append(game_field.step_turn)

			return {
				'l_d_name_to_df': l_d_name_to_df,
				'l_step_turn': l_step_turn,
			}

		print('Define new Function!')
		mult_proc_mng.define_new_func('func_play_some_games', play_some_games)
		print('Do the jobs!!')
		l_game_nr_all = [i for i in range(1, MAX_GAME_NR+1)]
		l_arguments = [(l_game_nr_all[x:x+CHUNK_SIZE], ) for x in range(0, MAX_GAME_NR, CHUNK_SIZE)]
		l_ret = mult_proc_mng.do_new_jobs(
			['func_play_some_games']*len(l_arguments),
			l_arguments,
		)
		print("len(l_ret): {}".format(len(l_ret)))
		# print("l_ret: {}".format(l_ret))

		# # testing the responsivness again!
		# mult_proc_mng.test_worker_threads_response()
		del mult_proc_mng

		l_d_name_to_df = []
		l_step_turn = []

		for d in l_ret:
			l_d_name_to_df.extend(d['l_d_name_to_df'])
			l_step_turn.extend(d['l_step_turn'])

		amount_step_turn = len(l_step_turn)
		arr_u, arr_c = np.unique(l_step_turn, return_counts=True)
		
		sum_prod_val = np.sum(arr_u * arr_c)
		estimation_step_turn = sum_prod_val / amount_step_turn

		print(f"amount_player: {amount_player}")
		print(f"AMOUNT_PIECE_PER_PLAYER: {AMOUNT_PIECE_PER_PLAYER}")
		print(f"AMOUNT_PIECE_FINISH_SIZE: {AMOUNT_PIECE_FINISH_SIZE}")
		print(f"AMOUNT_PIECE_FIELD_SIZE: {AMOUNT_PIECE_FIELD_SIZE}")
		print(f"l_player_color: {l_player_color}")
		print(f"l_player_color_idx: {l_player_color_idx}")

		print(f"amount_step_turn: {amount_step_turn}")
		print(f"sum_prod_val: {sum_prod_val}")
		print(f"estimation_step_turn: {estimation_step_turn}")

		def get_1d_data_stats(arr):
			if isinstance(arr, list):
				arr = np.array(arr, dtype=object)
			elif isinstance(arr, np.ndarray):
				pass
			else:
				globals()['arr_loc'] = arr
				assert False and "Need to be either a list or an array! (1d data)"

			assert len(arr.shape) == 1

			# TODO: add to remove NaN or None values too

			d = {}

			d['n'] = arr.shape[0]
			d['mean'] = np.mean(arr)
			d['std'] = np.std(arr)
			d['min'] = np.min(arr)
			d['median'] = np.median(arr)
			d['max'] = np.max(arr)

			return d

		df_player = pd.concat([d['df_player_stats'] for d in l_d_name_to_df])
		df_player_grpby = df_player.groupby(by=['player_color', 'player_color_idx']).apply(dict).reset_index().rename(mapper={0: 'dict'}, axis=1)

		# df_player_grpby['dict'].apply(lambda x: x['amount_dice_roll'].values)

		ser_amount_dice_roll = df_player_grpby['dict'].apply(lambda x: get_1d_data_stats(x['amount_dice_roll'].values))
		ser_amount_move = df_player_grpby['dict'].apply(lambda x: get_1d_data_stats(x['amount_move'].values))
		ser_place = df_player_grpby['dict'].apply(lambda x: get_1d_data_stats(x['place'].values))

		df_stats_avrg = pd.concat([
				df_player_grpby['player_color'],
				df_player_grpby['player_color_idx'],

				ser_amount_dice_roll.apply(lambda x: x['mean']).rename('amount_dice_roll_mean'),
				ser_amount_dice_roll.apply(lambda x: x['std']).rename('amount_dice_roll_std'),
				ser_amount_dice_roll.apply(lambda x: x['median']).rename('amount_dice_roll_median'),

				ser_amount_move.apply(lambda x: x['mean']).rename('amount_move_mean'),
				ser_amount_move.apply(lambda x: x['std']).rename('amount_move_std'),
				ser_amount_move.apply(lambda x: x['median']).rename('amount_move_median'),

				ser_place.apply(lambda x: x['mean']).rename('place_mean'),
				ser_place.apply(lambda x: x['std']).rename('place_std'),
				ser_place.apply(lambda x: x['median']).rename('place_median'),
			],
			axis=1,
		).sort_values(by=['player_color_idx']).reset_index(drop=True)

		str_player_color_idx = ','.join(list(map(str, l_player_color_idx)))
		df_stats_avrg['str_player_color_idx'] = str_player_color_idx
		
		df_stats_avrg['amount_piece_per_player'] = AMOUNT_PIECE_PER_PLAYER
		df_stats_avrg['amount_piece_finish_size'] = AMOUNT_PIECE_FINISH_SIZE
		df_stats_avrg['amount_piece_field_size'] = AMOUNT_PIECE_FIELD_SIZE

		l_column = df_stats_avrg.columns.tolist()
		l_column.insert(2, l_column.pop(l_column.index('str_player_color_idx')))
		l_column.insert(0, l_column.pop(l_column.index('amount_piece_per_player')))
		l_column.insert(0, l_column.pop(l_column.index('amount_piece_finish_size')))
		l_column.insert(0, l_column.pop(l_column.index('amount_piece_field_size')))
		df_stats_avrg = df_stats_avrg[l_column]

		print(f"df_stats_avrg:\n{df_stats_avrg}")

		l_df_stats_avrg.append(df_stats_avrg)

		# FIXME: change the priorities for each TODO if needed
		# TODO: make more functional style and oop style (do refactoring)
		# TODO: write function for the first interactions
		# TODO: write test cases later for testing the play game_field!

		# TODO: create the images for each tile programable with code, not hardcoded as image! (if possible)
