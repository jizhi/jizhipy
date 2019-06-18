
class ModHelpFormatter(object):

	def _format_action_invocation(self, action):
		''' from: argparse.py --> class HelpFormatter._format_actions_invocation() '''
		if not action.option_strings:
			metavar, = self._metavar_formatter(action, action.dest)(1)
			return metavar
		else:
			parts = []
			# if Optional doesn't take a value, format is:
			#	-s, --long
			if action.nargs == 0:
				parts.extend(action.option_strings)
			# if Optional takes a value, format is:
			#	-s ARGS, --long ARGS
			else:
				try: default = action.type.__name__[0].upper()
				except: default = 'X'
			#	default = action.dest.upper()
				args_string=self._format_args(action, default)
				for option_string in action.option_strings:
					parts.append('%s %s' % (option_string, args_string))
			return ', '.join(parts)


	def _format_actions_usage(self, actions, groups):
		''' from: argparse.py --> class HelpFormatter._format_actions_usage() '''
		import re as _re
		SUPPRESS = '==SUPPRESS=='
		# find group indices and identify actions in groups
		group_actions = set()
		inserts = {}
		for group in groups:
			try: start=actions.index(group._group_actions[0])
			except ValueError: continue
			else:
				end = start + len(group._group_actions)
				if actions[start:end] == group._group_actions:
					for action in group._group_actions:
						group_actions.add(action)
					if not group.required:
						if start in inserts:
							inserts[start] += ' ['
						else: inserts[start] = '['
						inserts[end] = ']'
					else:
						if start in inserts:
							inserts[start] += ' ('
						else: inserts[start] = '('
						inserts[end] = ')'
					for i in range(start + 1, end):
						inserts[i] = '|'
		# collect all actions format strings
		parts = []
		for i, action in enumerate(actions):
			# suppressed arguments are marked with None
			# remove | separators for suppressed arguments
			if action.help is SUPPRESS:
				parts.append(None)
				if inserts.get(i) =='|': inserts.pop(i)
				elif inserts.get(i+1) =='|': inserts.pop(i+1)
			# produce all arg strings
			elif not action.option_strings:
				part = self._format_args(action, action.dest)
				# if it's in a group, strip the outer []
				if action in group_actions:
					if part[0] == '[' and part[-1] == ']':
						part = part[1:-1]
				# add the action string to the list
				parts.append(part)
			# produce first way to invoke option in brackets
			else:
				option_string = action.option_strings[0]
				# if Optional doesn't take a value, format is:
				#	-s or --long
				if action.nargs == 0:
					part = '%s' % option_string
				# if Optional takes a value, format is:
				#	-s ARGS or --long ARGS
				else:
					try: default = action.type.__name__[0].upper()
					except: default = 'X'
					args_string = self._format_args(action, default)
					part='%s %s' % (option_string,args_string)

				# make it look optional if it's not required or in a group
				if not action.required and action not in group_actions:
					part = '[%s]' % part
				# add the action string to the list
				parts.append(part)
		# insert things at the necessary indices
		for i in sorted(inserts, reverse=True):
			parts[i:i] = [inserts[i]]
		# join all the action items with spaces
		text = ' '.join([item for item in parts if item is not None])
		# clean up separators for mutually exclusive groups
		open = r'[\[(]'
		close = r'[\])]'
		text = _re.sub(r'(%s) ' % open, r'\1', text)
		text = _re.sub(r' (%s)' % close, r'\1', text)
		text = _re.sub(r'%s *%s' % (open, close), r'', text)
		text = _re.sub(r'\(([^|]*)\)', r'\1', text)
		text = text.strip()
		# return the text
		return text





def ArgparseFormatter(indent_increment=2, max_help_position=30, width=None, formatter='ArgumentDefaultsHelpFormatter'):
	'''
	formatter:
		[str]
		'HelpFormatter'
		'ArgumentDefaultsHelpFormatter'
		'RawDescriptionHelpFormatter'
		'RawTextHelpFormatter'

	Usage:
		import argparse
		parser = argparse.ArgumentParser(description='ABC', formatter_class=jp.Basic.ArgparseFormatter())
		parser.add_argument('--image', '-i', default='', type=str, help='Path of an image')
		parser.add_argument('--stream', action='store_true', help='Read the video stream from a camera by opencv')
		args = parser.parse_args()
	'''
	import os
	from argparse import HelpFormatter, ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter, RawTextHelpFormatter
	try: width = int(width)
	except: width = int(os.popen('stty size').read()[:-1].split(' ')[-1])-2
	#########################################
#	class NewHelpFormatter(HelpFormatter, ModHelpFormatter):
	class NewHelpFormatter(ModHelpFormatter, HelpFormatter):
		def __init__(self, prog, indent_increment=indent_increment, max_help_position=max_help_position, width=width, **kwargs):
			HelpFormatter.__init__(self, prog, indent_increment, max_help_position, width, **kwargs)
	#########################################
	class NewArgumentDefaultsHelpFormatter(NewHelpFormatter, ArgumentDefaultsHelpFormatter): pass
	#----------------------------------------
#	class NewRawDescriptionHelpFormatter(RawDescriptionHelpFormatter, NewHelpFormatter): pass
	NewRawDescriptionHelpFormatter =RawDescriptionHelpFormatter
	#----------------------------------------
#	class NewRawTextHelpFormatter(NewRawDescriptionHelpFormatter): pass
	NewRawTextHelpFormatter = RawTextHelpFormatter
	#----------------------------------------
	if (formatter =='ArgumentDefaultsHelpFormatter'): return NewArgumentDefaultsHelpFormatter
	elif (formatter =='RawDescriptionHelpFormatter'): return NewRawDescriptionHelpFormatter
	elif (formatter =='RawTextHelpFormatter'): return NewRawTextHelpFormatter
	else: return NewHelpFormatter





class ArgumentParser(object):

	def __init__(self, description=None, formatter='ArgumentDefaultsHelpFormatter', **kwargs):
		'''
		Parameters
		----------
		**kwargs:
			ArgparseFormatter's parameters +
			argparse.ArgumentParser's parameters
		'''
		import argparse
		self.del_action, self.del_option = [], {}
		ArgumentParser_dict = {'description':description}
		ArgumentParser_name = ['prog', 'usage', 'description', 'epilog', 'version', 'parents', 'formatter_class', 'prefix_chars', 'conflict_handler', 'add_help']
		Formatter_dict = {'formatter':formatter}
		Formatter_name = ['indent_increment', 'max_help_position', 'width', 'formatter']
		#----------------------------------------
		for k in ArgumentParser_name:
			if (k not in kwargs.keys()): continue
			ArgumentParser_dict[k] = kwargs[k]
		for k in Formatter_name:
			if (k not in kwargs.keys()): continue
			Formatter_dict[k] = kwargs[k]
		ArgumentParser_dict['formatter_class'] = ArgparseFormatter(**Formatter_dict)
		self.parser = argparse.ArgumentParser(**ArgumentParser_dict)





	def Add(self, *args, **kwargs):
		'''
		Parameters
		----------
		*args:
			(1) short_option
			(2) long_option
			(3) (short_option, long_option)

		**kwargs:
			(1) True/False: 
					action='store_true' | 'store_false'
			(2) value:
					default=...   type=...
			help=
		'''
		self.parser.add_argument(*args, **kwargs)





	def Remove(self, *args):
		'''
		Parameters
		----------
		*args:
			long_option | short_option
			e.g. --max    -m
		'''
		n, N = 0, len(self.parser.__dict__['_actions'])
		deal_keys = []
		for i in range(N):
			s, d = self.parser.__dict__['_actions'][i-n].option_strings, False
			for j in s:
				if (j in args): 
					d = True
					break
			if (d):
				self.del_action.append(self.parser.__dict__['_actions'][i-n])
				self.parser.__dict__['_actions'].pop(i-n)
				deal_keys.append(s)
				n += 1
		for k in deal_keys: 
			for j in k:
				self.del_option[j] = self.parser.__dict__['_option_string_actions'][j]
				self.parser.__dict__['_option_string_actions'].pop(j)





	def Recover(self, *args):
		'''
		Parameters
		----------
		*args:
			long_option | short_option
			e.g. --max    -m
		'''
		n, N = 0, len(self.del_action)
		deal_keys = []
		for i in range(N):
			s, d = self.del_action[i-n].option_strings, False
			for j in s:
				if (j in args): 
					d = True
					break
			if (d):
				deal_keys.append(s)
				self.parser.__dict__['_actions'].append(self.del_action[i-n])
				self.del_action.pop(i-n)
				n += 1
		for k in deal_keys: 
			for j in k:
				self.parser.__dict__['_option_string_actions'][j] = self.del_option[j]
				self.del_option.pop(j)





	def Get(self):
		return self.parser.parse_args()

