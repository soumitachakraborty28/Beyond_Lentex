import importlib.util
mods=['numpy','pandas','sklearn','sentence_transformers','transformers','torch','tqdm','yaml']
for m in mods:
    print(f'{m}:', 'OK' if importlib.util.find_spec(m) else 'MISSING')
