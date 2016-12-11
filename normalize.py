import learn

dist = learn.read_dist("sampled_color_full.mat")
new_dist = learn.normalize(dist)
learn.write_dist(new_dist,"sampled_color_normalized.mat")
