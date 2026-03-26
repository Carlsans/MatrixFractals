class Configuration:
    def __init__(self):

        # GA
        self.poolsize = 200# -45
        self.mutation_probability = 0
        self.crossing_probability = 1
        self.row_shift_probability = 0.1
        self.mutation_flip_probability = 0.001 # Apply to all items each turn

        # 3D rendering
        self.eye_disparity=.16
        self.accumulate_rays = True
        self.max_iter_render = 100
        rendermode = "render"
        if rendermode == "test":
            self.video_length = 10
            self.fps = 10
            self.out_W = 256

        if rendermode == "middle":
            self.video_length = 20
            self.fps = 20
            self.out_W = 1000

        if rendermode == "render":
            self.video_length = 60
            self.fps = 20
            self.out_W = 2528
        #self.out_W = 512