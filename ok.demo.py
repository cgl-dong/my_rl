memory = [
    [
        {'cur_phase': [0, 1, 0, 1, 0, 0, 0, 0],
            'coming_vehicle': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0],
            'leaving_vehicle': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0]}, 0,
        {'cur_phase': [0, 1, 0, 1, 0, 0, 0, 0],
                                                           'coming_vehicle': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1,
                                                                              0, 0, 0, 0, 0, 0, 0, 0],
                                                           'leaving_vehicle': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                               0, 0, 0, 0, 0, 0, 0, 0]}, 0.0, 0.0, 0,
           'generator_0-round_0'],
    [
        {'cur_phase': [0, 1, 0, 1, 0, 0, 0, 0],
                                     'coming_vehicle': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                        0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                     'leaving_vehicle': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, 3,
                                    {'cur_phase': [0, 0, 0, 0, 1, 0, 1, 0],
                                     'coming_vehicle': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                        2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                     'leaving_vehicle': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}]]

res = []

for i in memory:
    temp = []
    for k in i:
        if isinstance(k, dict):
            temp.extend(k.get("coming_vehicle"))
    res.append(temp)

print(len(res[0]))
