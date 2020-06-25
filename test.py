import unittest
import logging
import numpy as np
import os


class TestNumPy(unittest.TestCase):

    def test_create_basic_arrays(self):
        int_array = np.array([1, 2, 3, 4, 5])

        print('Dimensione:', int_array.shape)
        print('Oggetto di tipo:', type(int_array))
        print('Elementi di tipo:', int_array.dtype)
        print('Elementi totali:', int_array.size)

        _array = np.array([1.5, 2.2, 3.7, 4.0, 5.9], dtype = np.int64)
        print('Elementi di tipo:', _array.dtype)

    def test_save_array(self):
        x = np.array([1, 2, 3, 4, 5])
        np.save('my_array', x)
        self.assertTrue(os.path.exists('my_array.npy'))

        y = np.load('my_array.npy')
        print('y = ', y)

    def test_array_velocity(self):
        import time

        random_array = np.random.random(100000000)

        start = time.time()
        media = sum(random_array) / len(random_array)
        print("Media {} in {}".format(media, time.time() - start))

        start = time.time()
        media2 = np.mean(random_array)
        print("Media {} in {}".format(media2, time.time() - start))

    def test_builtin_function(self):
        zero_array = np.zeros((3,4))
        print('zero_array = ', zero_array)
        ones_array = np.ones((3,2))
        print('ones_array = ', ones_array)
        full_array = np.full((2,3), 5)
        print('full_array = ', full_array)
        identity_array = np.eye(5)
        print('identity_array = ', identity_array)
        diagonal_array = np.diag([10,20,30,50])
        print('diagonal_array = ', diagonal_array)

        # --------------------
        x = np.arange(10)
        x = np.arange(4,10)
        x = np.arange(1,14,3)
        x = np.linspace(0,25,10)
        x = np.linspace(0,25,10, endpoint = False)

        # --------------------
        Y = np.arange(20).reshape(4, 5)
        random_float = np.random.random((3,3))
        random_int = np.random.randint(4,15,size=(3,2))

        # Creazione di una array gaussiana con media 0 e deviazione standard 0.1
        gaussian_array = np.random.normal(0, 0.1, size=(1000,1000))
        print('Dimensione:', gaussian_array.shape)
        print('Oggetto di tipo:', type(gaussian_array))
        print('Elementi di tipo:', gaussian_array.dtype)
        print('Media:', gaussian_array.mean())
        print('max value:', gaussian_array.max())
        print('min value:', gaussian_array.min())
        print('Numeri totali negativi:', (gaussian_array < 0).sum())
        print('Numeri totali positivi:', (gaussian_array > 0).sum())

    def test_lista_numeri_pari(self):
        arrange_array = np.arange(2,42,2).reshape(5,4)
        linspace_array = np.linspace(2,40,20).reshape(5,4)

    def test_modifica_arrays(self):
        # Array di rango 1
        array_1 = np.array([1, 2, 3, 4, 5])

        print('Primo elemento:', array_1[0]) 
        print('Ultimo elemento:', array_1[-1])

        # Array di rango 2
        array_2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
        print(array_2)
        print('Primo elemento:', array_2[0][0]) 
        print('Ultimo elemento:', array_2[-1][-1])

        # Cancelliamo posizioni 0 e 4
        plain_array = np.delete(array_2, [0,4])
        print('Nuova array:', plain_array)

        # Canceliamo riga
        deleted_riga = np.delete(array_2, 0, axis=0)
        print('Array senza riga:', deleted_riga)

        # Canceliamo colonne
        deleted_colonna = np.delete(array_2, [0,2], axis=1)
        print('Array senza colonne:', deleted_colonna)

        # Aggiungere elementi:
        x = np.append(array_1, 6)
        print('Array 1 con nuovi elementi:', x)
        x = np.append(array_2, [[10,11,12]], axis=0)
        print('Array 2 con nuovi elementi (x):', x)
        y = np.append(array_2, [[10],[11], [12]], axis=1)
        print('Array 2 con nuovi elementi (y):', y)

        # Inseriamo i valori 20 e 40 entro il 2 e 5
        array_inserita = np.insert(array_1, 2, [20, 40])
        print('Array 1 inseriti elementi:', array_inserita)

        # Inseriamo una riga dopo la prima (posizione 1)
        w = np.insert(array_2, 1, [20,30,40], axis=0)
        print('Array 2 inseriti elementi (x):', x)
        w = np.insert(array_2, 1, [20,30,40], axis=1)
        print('Array 2 inseriti elementi (y):', y)

        # Funzioni stack
        array_rank1_2 = np.array([1,2])

        array_rank2 = np.array([[3,4],[5,6]])
        print(array_rank2)

        z = np.vstack((array_rank1_2, array_rank2))
        print(z)

        w = np.hstack((array_rank2, array_rank1_2.reshape(2,1)))
        print(w)

    def test_slice_arrays(self):
        # TODO
        pass
        
    def test_select_arrays(self):
        # We create a 5 x 5 ndarray that contains integers from 0 to 24
        base_array = np.arange(25).reshape(5, 5)

        # We use Boolean indexing to select elements in X:
        print('Elementi maggiori di 10:', base_array[base_array > 10])
        print('Elementi maggiori minori o uguali a 7:', base_array[base_array <= 7])
        print('Elementi entre 10 e 17:', base_array[(base_array > 10) & (base_array < 17)])

        # We use Boolean indexing to assign the elements that are between 10 and 17 the value of -1
        base_array[(base_array > 10) & (base_array < 17)] = -1

        # We create a rank 1 ndarray
        array_1 = np.array([1,2,3,4,5])
        print(array_1)

        # We create a rank 1 ndarray
        array_2 = np.array([6,7,2,8,4])
        print(array_2)

        print('The elements that are both in x and y:', np.intersect1d(array_1,array_2))
        print('The elements that are in x that are not in y:', np.setdiff1d(array_1,array_2))
        print('Tutti gli elementi:',np.union1d(array_1,array_2))

    def test_ordenamento_arrays(self):
        base_array = np.random.randint(1,11,size=(10,))

        print('Ordinamento con return:', np.sort(base_array))
        print('Ordinamento unico:', np.sort(np.unique(base_array)))

        base_array.sort()
        print('Ordinamento nell\'array:', base_array)

        base_array2 = np.random.randint(1,11,size=(5,5))
        print('Ordinata per colonne:\n', np.sort(X, axis = 0))
        print('Ordinata per righe:\n', np.sort(X, axis = 1))
  
    def test_operazioni_aritmetiche(self):
        array_1 = np.array([1,2,3,4])
        array_2 = np.array([5.5,6.5,7.5,8.5])

        print('array_1 + array_2 = ', array_1 + array_2)
        print('add(array_1,y) = ', np.add(array_1,y))
        print()
        print('array_1 - array_2 = ', array_1 - array_2)
        print('subtract(array_1,y) = ', np.subtract(array_1,y))
        print()
        print('array_1 * array_2 = ', array_1 * array_2)
        print('multiply(array_1,array_2) = ', np.multiply(array_1,array_2))
        print()
        print('array_1 / array_2 = ', array_1 / array_2)
        print('divide(array_1,y) = ', np.divide(array_1,array_2))

        X = np.array([1,2,3,4]).reshape(2,2)
        Y = np.array([5.5,6.5,7.5,8.5]).reshape(2,2)

        print('X + Y = \n', X + Y)
        print()
        print('add(X,Y) = \n', np.add(X,Y))
        print()
        print('X - Y = \n', X - Y)
        print()
        print('subtract(X,Y) = \n', np.subtract(X,Y))
        print()
        print('X * Y = \n', X * Y)
        print()
        print('multiply(X,Y) = \n', np.multiply(X,Y))
        print()
        print('X / Y = \n', X / Y)
        print()
        print('divide(X,Y) = \n', np.divide(X,Y))

        x = np.array([1,2,3,4])

        print()
        print('EXP(x) =', np.exp(x))
        print()
        print('SQRT(x) =',np.sqrt(x))
        print()
        print('POW(x,2) =',np.power(x,2)) # We raise all elements to the power of 2

        X = np.array([[1,2], [3,4]])
        print('Average of all elements in X:', X.mean())
        print('Average of all elements in the columns of X:', X.mean(axis=0))
        print('Average of all elements in the rows of X:', X.mean(axis=1))
        print()
        print('Sum of all elements in X:', X.sum())
        print('Sum of all elements in the columns of X:', X.sum(axis=0))
        print('Sum of all elements in the rows of X:', X.sum(axis=1))
        print()
        print('Standard Deviation of all elements in X:', X.std())
        print('Standard Deviation of all elements in the columns of X:', X.std(axis=0))
        print('Standard Deviation of all elements in the rows of X:', X.std(axis=1))
        print()
        print('Median of all elements in X:', np.median(X))
        print('Median of all elements in the columns of X:', np.median(X,axis=0))
        print('Median of all elements in the rows of X:', np.median(X,axis=1))
        print()
        print('Maximum value of all elements in X:', X.max())
        print('Maximum value of all elements in the columns of X:', X.max(axis=0))
        print('Maximum value of all elements in the rows of X:', X.max(axis=1))
        print()
        print('Minimum value of all elements in X:', X.min())
        print('Minimum value of all elements in the columns of X:', X.min(axis=0))
        print('Minimum value of all elements in the rows of X:', X.min(axis=1))