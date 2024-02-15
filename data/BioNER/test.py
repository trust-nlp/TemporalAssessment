import re

def is_sentence_end(prev_token, token, next_token):
    if token in {'!', '?'}:
        return True
    if token == '.':
        # if there are no next token，or next token is capitalized
        if not next_token or re.match(r'^[A-Z]', next_token):
            if not re.match(r'^[A-Z]$', prev_token): # For abbreviation like name M.M.Lightfoote
                return True  
        else:
            return False 
    return False  
#'Dr','.','Bush'  
# M.M.Light
print(is_sentence_end("1", ".", "Our"))
print(is_sentence_end("monocytes", ".", "In"))