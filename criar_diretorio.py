import numpy as np
import os

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['a', 'b', 'c', 'd',
                    'e', 'f', 'g', 'h',
                    'i', 'j', 'k', 'l',
                    'm', 'n', 'o', 'p', 
                    'q', 'r', 's', 't',
                    'u', 'v', 'w', 'x',
                    'y', 'z', 'Oi', 'Obrigado',
                    'Te Amo', 'Tchau', 'sim', 'não',
                    'Casa'])
no_sequences = 30
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass