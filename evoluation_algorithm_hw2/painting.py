#   EE496 HOMEWORK II - Evolutionary Art
#   Halil Berk Dergi - 2093649


import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

num_generations = 20000
num_inds = 20
num_genes = 50
tm_size = 5
frac_elites = 0.2
frac_parents = 0.6
mutation_prob = 0.2
mutation_type = 'guided'

img = cv2.imread('painting.png')

height, width, channels = img.shape


def fitness(image):
# to overcome overflow problem, they are defined in int64 format
    pixel_src = np.int64(img)
    pixel_image = np.int64(image)
    image_diffirence = np.subtract(pixel_src, pixel_image)
    #fitness_ind =-np.sum(np.square(np.subtract(pixel_src, pixel_image)))
    fitness_ind = -np.linalg.norm(image_diffirence)

    return fitness_ind


def selection(fitness_gen):
    fitness_index = np.empty([len(fitness_gen), 1])
    index_list = []
    temp_array = np.empty(len(fitness_gen))
    for k in range(0,len(fitness_gen)):
        temp_array[k] = k

#Select the elitist
    for i in range(0, int(frac_elites * len(fitness_gen))): # elite sayısı kadar
        best = int(np.random.choice(temp_array, 1))
        for j in range(0, tm_size):
            ind = int(np.random.choice(temp_array, 1))
            if fitness_gen[ind] > fitness_gen[best]:
                best = ind
        temp_index = np.where(best==temp_array)
        temp_array = np.delete(temp_array, temp_index, axis=0)
        #for t in range(best, len(temp_array)):
        #    temp_array[t] = t-1

        index_list.append(best)
        #print(temp_array)
    print(index_list)
    return index_list

def mutation(inds_copy):
    x = random.uniform(0, 1)

# Check whether mutation will occur or not
    if x < mutation_prob:
        i = random.randint(0, len(inds_copy)-1)
        j = random.randint(0, num_genes-1)
        if mutation_type == 'guided':
            inds_copy[i, j, 0] = random.randint(inds_copy[i, j, 0] - int(width / 4), inds_copy[i, j, 0] + int(width / 4))
            if inds_copy[i, j, 0] < 0:
                inds_copy[i, j, 0] = 0
            elif inds_copy[i, j, 0] > width:
                inds_copy[i, j, 0] = width
            inds_copy[i, j, 1] = random.randint(inds_copy[i, j, 1] - int(height / 4), inds_copy[i, j, 1] + int(height / 4))
            if inds_copy[i, j, 1] < 0:
                inds_copy[i, j, 1] = 0
            elif inds_copy[i, j, 1] > height:
                inds_copy[i, j, 1] = height
            inds_copy[i, j, 2] = random.randint(inds_copy[i, j, 2] - 10, inds_copy[i, j, 2] + 10)
            if inds_copy[i, j, 2] < 0:
                 inds_copy[i, j, 2] = 0
            elif inds_copy[i, j, 2] > width:
                inds_copy[i, j, 2] = width
            inds_copy[i, j, 3] = random.randint(inds_copy[i, j, 3] - 64, inds_copy[i, j, 3] + 64)
            if inds_copy[i, j, 3] < 0:
                inds_copy[i, j, 3] = 0
            elif inds_copy[i, j, 3] > 255:
                inds_copy[i, j, 3] = 255
            inds_copy[i, j, 4] = random.randint(inds_copy[i, j, 4] - 64, inds_copy[i, j, 4] + 64)
            if inds_copy[i, j, 4] < 0:
                inds_copy[i, j, 4] = 0
            elif inds_copy[i, j, 4] > 255:
                inds_copy[i, j, 4] = 255
            inds_copy[i, j, 5] = random.randint(inds_copy[i, j, 5] - 64, inds_copy[i, j, 5] + 64)
            if inds_copy[i, j, 5] < 0:
                inds_copy[i, j, 5] = 0
            elif inds_copy[i, j, 5] > 255:
                inds_copy[i, j, 5] = 255
            inds_copy[i, j, 6] = random.uniform(inds_copy[i, j, 6] - 0.25, inds_copy[i, j, 6] + 0.25)
            if inds_copy[i, j, 6] < 0:
                inds_copy[i, j, 6] = 0
            elif inds_copy[i, j, 6] > 1:
                inds_copy[i, j, 6] = 1
        else:
            for i in range(0,len(inds_copy)):
                for j in range(0,num_genes):
                    inds_copy[i, j, 0] = random.randint(0, height)
                    inds_copy[i, j, 1] = random.randint(0, width)
                    inds_copy[i, j, 2] = random.randint(0, width)
                    inds_copy[i, j, 3] = random.randint(0, 255)
                    inds_copy[i, j, 4] = random.randint(0, 255)
                    inds_copy[i, j, 5] = random.randint(0, 255)
                    inds_copy[i, j, 6] = random.uniform(0, 1)
    return inds_copy

def evaluation(inds, draw, d):
    fitness_array = np.zeros([num_inds])
    # Initialize < image > completely white with the same shape as the < source_image >.
    for i in range(0, num_inds):
        rgb_color = (255, 255, 255)
        image = np.zeros((height, width, channels), np.uint8)
        image[:] = rgb_color
        image_2 = image

# Draw circles
        for j in range(0, num_genes):
            cv2.circle(image_2, center=(int(inds[i, j, 0]), int(inds[i, j, 1])), radius=int(inds[i, j, 2]),
                       color=(int(inds[i, j, 3]), int(inds[i, j, 4]), int(inds[i, j, 5])), thickness=-1)
            image = cv2.addWeighted(image_2, inds[i, j, 6], image, 1-inds[i, j, 6], 0)

        fitness_array[i] = fitness(image)
    #print(fitness_array)

    index_image = np.where(fitness_array == np.amax(fitness_array))
    index_image = index_image[0][0]
    print(index_image)

    if draw == 'true':
        rgb_color = (255, 255, 255)
        image = np.zeros((height, width, channels), np.uint8)
        image[:] = rgb_color
        image_2 = image
        for j in range(0, num_genes):
            cv2.circle(image_2, center=(int(inds[index_image, j, 0]), int(inds[index_image, j, 1])), radius=int(inds[index_image, j, 2]),
                       color=(int(inds[index_image, j, 3]), int(inds[index_image, j, 4]), int(inds[index_image, j, 5])), thickness=-1)
            image = cv2.addWeighted(image_2, inds[index_image, j, 6], image, 1 - inds[index_image, j, 6], 0)



# Save the drawn image to the directory in PNG format
        cv2.imwrite(str(d) + '.png', image)

    return fitness_array

def crossover(fitness_generation_copy, inds_copy):

    fitness_index = np.argsort(fitness_generation_copy)
    parents_list_index = []
    childs = []
    child1=[]
    child2=[]
    delete_index = []
    for i in range(0, int(frac_parents*num_inds)):
        parents_list_index.append(fitness_index[-i-1])
    #print(parents_list_index)

# Apply one point crossover to the parents
    for j in range(int(len(parents_list_index)/2)):
        parent1 = inds_copy[parents_list_index[2*j]] # 1 invidual 20 gene var
        parent2 = inds_copy[parents_list_index[j*2+1]]
        delete_index.append(2*j)
        delete_index.append(2*j+1)
        crossover_point = random.randint(0, num_genes-1)
        gen1, gen2 = np.split(parent1, [crossover_point])
        gen3, gen4 = np.split(parent2, [crossover_point])

# Crossed genes are placed to the springoff
        child1 = np.concatenate((gen1, gen4), axis=0)
        child2 = np.concatenate((gen3, gen2), axis=0)
        childs.append(child1)
        childs.append(child2)

    if np.mod(len(parents_list_index), 2) == 0:
        inds_copy = np.delete(inds_copy, [delete_index], axis=0)
    else:
        inds_copy = inds_copy[-1]
    return childs, inds_copy

def main():

# Initialize population with <num_inds> individuals each having <num_genes> genes
    dtype = [('x', int), ('y', int), ('radius', int), ('r', int), ('g', int), ('b', int), ('alpha', float)] # Each gene has at least 7 values
    inds = np.zeros([num_inds, num_genes, 7])    #creating individuals (each individual has one chromosome)
    genes = np.zeros((1, 7)) #7 genes in a chromosome
    d = 0
    fitness_generation = np.empty([num_generations, num_inds]) #fitness of all invidiuals
    final_fitness = np.empty([num_generations])
    elite_inds = np.empty([int(frac_elites*num_inds)]) #elites

    for i in range(0, num_inds):
        for j in range(0, num_genes):
            genes[0, 0] = random.randint(0, 2*width)  # x coordinate
            genes[0, 1] = random.randint(0, 2*height) # y cordinate
            genes[0, 2] = random.randint(0, 2*width)  # radius
            # if a circle is not within the image (it lies outside completely), the corresponding gene should be reinitialized randomly until it is.
            while (genes[0, 0] - genes[0, 2] > width) | (genes[0, 1] - genes[0, 2] > height):
                genes[0, 0] = random.randint(0, height)   #x coordinate
                genes[0, 1] = random.randint(0, width)    #y cordinate
                genes[0, 2] = random.randint(0, width)    #radius
            genes[0, 3] = random.randint(0, 255)
            genes[0, 4] = random.randint(0, 255)
            genes[0, 5] = random.randint(0, 255)
            genes[0, 6] = random.uniform(0, 1)
            inds[i, j] = genes

# Sort the genes
    for l in range(0,num_inds):
        a = sorted(inds[l], key=lambda x: x[2], reverse=True)
        a = np.asarray(a)
        inds[l] = a
# Continue until all generations are done
    for k in range(0, num_generations):
        if np.mod(k, 1000) == 0:
            draw = 'true'
        elif k == 19999:
            draw = 'true'
        else:
            draw = 'false'
        print('Generation ' + str(k) + ' is done')

# Evaluate generations
        fitness_generation[k] = evaluation(inds, draw, d)
        draw = 'false'
        d = d + 1
        fitness_generation_copy = fitness_generation[k]


# Select elit individuals then forward them directly to the next generation

        elite_inds = selection(fitness_generation[k])
        inds_copy = np.delete(inds, elite_inds, axis=0)
        fitness_generation_copy = np.delete(fitness_generation_copy, elite_inds, axis=0)
        #print('num 234copy_inds ', len(inds_copy[:, 0]))

# Apply crossover method to each parent except elitist(s)
        childs, inds_copy = crossover(fitness_generation_copy, inds_copy)
        childs = np.asarray(childs)
        #print('num 238copy_inds ', len(inds_copy[:, 0]))
        #print('num childs', len(childs[:, 0]))
        try:
            inds_copy = np.concatenate((childs, inds_copy), axis=0)
        except ValueError:
            inds_copy = childs
        #print('num elite',len(inds[elite_inds]))


# Mutation is done on each individual
        inds_copy = mutation(inds_copy)
        inds = np.concatenate((inds[elite_inds], inds_copy), axis=0)
        print('num inds',len(inds[:,0]))
        fitness_index = np.where(fitness_generation[k] == np.amax(fitness_generation[k]))
        final_fitness[k] = fitness_generation[k, fitness_index[0][0]]
        print('fitness : ', final_fitness[k])



# Plot from generation 1 to generation 10000
    plt.figure()
    plt.plot(final_fitness)
    plt.savefig('fitness_plot_1_to_10000.png')

# Plot from generation 1000 to generation 10000
    plt.figure()
    plt.plot(final_fitness[999:19999])
    plt.savefig('fitness_plot_1000_to_10000.png')


if __name__ == "__main__":
    main()