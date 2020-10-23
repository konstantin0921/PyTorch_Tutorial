from tqdm import tqdm, trange
import time

iterator = tqdm([1, 2, 3, 4, 5])  # class tqdm.std.tqdm

print(type(iterator))

for i in iterator:
    time.sleep(0.3)
    iterator.set_postfix(whatever=float(i))


for i in trange(10):
    time.sleep(0.3)


with tqdm(total=100) as pbar:
    for i in range(10):
        time.sleep(0.3)
        pbar.update(10)
