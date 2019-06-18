
def Keyboard2Int(key=None):
	'''
	What is the ASCII of a key on the keyboard?

	Parameters
	----------
	key:
		(1) [str], a key on the keyboard. For example: 'esc', '>'/'.'/'.>'/'>.'
		(2) None: return a dict
		** Enter1 is the Enter button on the function panel
		** Enter2 is the Enter button on the number panel

	Returns
	----------
	a dict or int
	'''
	code = {'A':65, 'B':66, 'C':67, 'D':68, 'E':69, 'F':70, 
	'G':71, 'H':72, 'I':73, 'J':74, 'K':75, 'L':76, 'M':77, 
	'N':78, 'O':79, 'P':80, 'Q':81, 'R':82, 'S':83, 'T':84, 
	'U':85, 'V':86, 'W':87, 'X':88, 'Y':89, 'Z':90, 
	'0)':48, '1!':49, '2@':50, '3#':51, '4$':52, '5%':53, 
	'6^':54, '7&':55, '8*':56, '9(':57, 
	'0':96, '1':97, '2':98, '3':99, '4':100, '5':101, 
	'6':102, '7':103, '8':104, '9':105, '*':106, 
	'+':107, 'Enter2':108, '-':109, '.':110, '/':111,
	'F1':112, 'F2':113, 'F3':114, 'F4':115, 'F5':116, 
	'F6':117, 'F7':118, 'F8':119, 'F9':120, 'F10':121, 
	'F11':122, 'F12':123, 'F13':124, 'F14':125, 'F15':126, 
	'Backspace':8, 'Tab':9, 'Enter1':13, 'Shift':16, 
	'Ctrl':17, 'Caps Lock':20, 'Esc':27, 'Space':32, 
	'Page Up':33, 'Page Down':34, 'End':35, 'Home':36, 
	'Left Arrow':37, 'Up Arrow':38, 'Right Arrow':39, 
	'Down Arrow':40, 'Insert':45, 'Delete':46, 
	'Num Lock':144, ';:':186, '=+':187, ',<':188, 
	'-_':189, '.>':190, '/?':191, '`~':192, '[{':219, 
	'\|':220, ']}':221, "'":222, '"':222}
	if (key is None): return code
	keys, values = list(code.keys()), list(code.values())
	key = str(key).lower()
	for i in range(len(keys)):
		if (key == keys[i].lower()): return values[i]
	

