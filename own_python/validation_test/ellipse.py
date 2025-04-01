import numpy as np
import matplotlib.pyplot as plt

def parabole(x):

    z = np.zeros_like(x)
    for i in range(len(x)):

        if x[i] < 8:
            z[i] = 0

        elif x[i] > 12:
            z[i] = 0

        else:
            z[i] = 0.2 - 0.05*(x[i]-10)**2

    return z
def main():
    x = np.linspace(8, 12, 1000)
    z = parabole(x)


    print(z)
    new_x = np.linspace(0, 4, len(z))
    plt.figure()
    plt.plot(new_x, z)
    plt.show()

if __name__=="__main__":
    main()