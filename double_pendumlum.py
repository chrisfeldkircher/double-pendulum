import pygame
from pygame import *
import sys
import numpy as np
from numpy.linalg import inv

#Parameters
m1, m2 = 1.0, 1.0
l1, l2 = 1.0, 1.0
a1, a2 = 0, 0
g = 9.81
offset = (400,200)
prev = None
t = 0.0
delta_t = 0.015
y = np.array([0.0, 0.0, np.pi-0.01, np.pi]) #a1_t, a2_t, a1, a2

def G(y,t):
	a1_t, a2_t = y[0], y[1]
	a1, a2 = y[2], y[3]
	c = np.array([[.009,0],[0,.009]]) #damping matrix

	m11, m12 = (m1+m2)*l1, m2*l2*np.cos(a1-a2)
	m21, m22 = l1*np.cos(a1-a2), l2
	m = np.array([[m11, m12],[m21, m22]]) #Equation of motion --> Euler-Lagrange

	f1 = -m2*l2*a2_t*a2_t*np.sin(a1-a2) - (m1+m2)*g*np.sin(a1)
	f1 = f1 - (c[0][0]*a1_t + c[0][1]*a2_t) #c*v = damping_force

	f2 = l1*a1_t*a1_t*np.sin(a1-a2) - g*np.sin(a2)
	f2 = f2 - (c[1][0]*a1_t + c[1][1]*a2_t)

	f = np.array([f1, f2])

	accel = inv(m).dot(f)

	return np.array([accel[0], accel[1], a1_t, a2_t])

def RK4_step(y, t, dt): #Runge-Kutta
	k1 = G(y,t)
	k2 = G(y+0.5*k1*dt, t+0.5*dt)
	k3 = G(y+0.5*k2*dt, t+0.5*dt)
	k4 = G(y+k3*dt, t+dt)

	return dt * (k1 + 2*k2 + 2*k3 + k4) /6

def polar_to_cartesian(a1,a2):
    scale = 100
    x1 =  scale*l1 * np.sin(a1) + offset[0]
    y1 =  scale*l1 * np.cos(a1) + offset[1]

    x2 = x1 + scale*l2 * np.sin(a2)
    y2 = y1 + scale*l2 * np.cos(a2)

    print(a1, a2)

    return (x1,y1),(x2,y2)

def draw(point1, point2):
    scale = 6
    x1, y1 = point1[0],point1[1]
    x2, y2 = point2[0],point2[1]

    if prev:
        x_l, y_l = prev[0], prev[1]
        pygame.draw.line(trace, GOLD, (x_l, y_l), (x2,y2), 2)

    screen.fill(WHITE)
    screen.blit(trace, (0,0))

    pygame.draw.line(screen, BLUE, offset,(x1,y1), 5)
    pygame.draw.line(screen, BLUE, (x1,y1),(x2,y2), 5)
    pygame.draw.circle(screen, BLUE, offset, 4)
    pygame.draw.circle(screen, BLUE, (x1,y1), m1*scale)
    pygame.draw.circle(screen, BLUE, (x2,y2), m2*scale)

    return (x2, y2)

#Configuration of the Screen
w,h = 800,480
runnin = True

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GOLD = (171, 130, 19, 50)
BLUE = (70,130,180)

screen = pygame.display.set_mode((w,h))
screen.fill(WHITE)
trace = screen.copy()
pygame.display.update()
clock = pygame.time.Clock()

while runnin:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    point1, point2 = polar_to_cartesian(y[2], y[3]) #y[0] = a1_t, y[1] = a2_t, y[2] = a1, y[3] = a2,
    prev = draw(point1, point2)

    t += delta_t
    y = y + RK4_step(y, t, delta_t)

    draw(point1, point2)

    clock.tick(60)
    pygame.display.update()
