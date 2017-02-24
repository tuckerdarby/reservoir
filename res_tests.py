from reservoir import Reservoir


res = Reservoir(width=5, length=7, height=1)
res.set_fluid(1, 0.25, 4000)
res.set_media((2, 2, 1), (1, 1, 0.4), 0.08)
res.create_well(2, 2, 0, 0, 100, flow=-5.615*11, pressure=500)
res.create_well(3, 4, 0, 0, 100, flow=5.615*10, pressure=500)
res.run_steps(1000, 0.1)  # 2231


