#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import csv

throttle_csvfile='throttles.csv'
steer_csvfile='steers.csv'
brakes_csvfile='brakes.csv'

# Empty placeholders for values
t_act = list()
t_prp = list()
s_act = list()
s_prp = list()
b_act = list()
b_prp = list()

print("Reading csv files")
with open(throttle_csvfile, 'r') as t_csv:
    for row in csv.DictReader(t_csv):
        t_act.append(float(row['actual']))
        t_prp.append(float(row['proposed']))
print("Finished reading {} entries from throttles csvfile".format(len(t_act)))

with open(steer_csvfile, 'r') as s_csv:
    for row in csv.DictReader(s_csv):
        s_act.append(float(row['actual']))
        s_prp.append(float(row['proposed']))
print("Finished reading {} entries from steers csvfile".format(len(s_act)))

with open(brakes_csvfile, 'r') as b_csv:
    for row in csv.DictReader(b_csv):
        b_act.append(float(row['actual']))
        b_prp.append(float(row['proposed']))
print("Finished reading {} entries from brakes csvfile".format(len(b_act)))

# print throttle values
for i in range(100,1000,25):
    print(t_act[i], t_prp[i])

print("Plotting the observation")

fig = plt.figure()
ax0=fig.add_subplot(311)
ax0.set_title("Throttle Values")
ax0.plot(range(len(t_act)), t_act, 'g.', label='actual')
ax0.plot(range(len(t_prp)), t_prp, 'b.', label='proposed')
ax0.legend(loc='upper right')
ax1=fig.add_subplot(312)
ax1.set_title("Steering Values")
ax1.plot(range(len(s_act)), s_act, 'g.', label='actual')
ax1.plot(range(len(s_prp)), s_prp, 'b.', label='proposed')
ax1.legend(loc='upper right')
ax2=fig.add_subplot(313)
ax2.set_title("Brakes Values")
ax2.plot(range(len(b_act)), b_act, 'g.', label='actual')
ax2.plot(range(len(b_prp)), b_prp, 'b.', label='proposed')
ax2.legend(loc='upper right')
plt.show()
