#%% Màximo
def maximo(a,b):
    if a > b:
        return a
    else:
        return b
    
#%% 2 en lista
def dos_pertenece(lista):
    return 2 in lista

#%% Pertenencia
def pertenece(lista, elem):
    return elem in lista

#%% Paridad
def es_par(n):
    return n%2 == 0

#%% La màs larga
def mas_larga(lista1, lista2):
    if len(lista1) > len(lista2):
        return lista1
    else:
        return lista2

#%% Tachar pares
def tachar_pares(lista):
    for i in range(len(lista)):
        if lista[i]%2 == 0:
            lista[i] = 'X'
    return lista

#%% Cantidad de e's
def cant_e(lista):
    cant = 0
    for elem in lista:
        if elem == 'e':
            cant+=1
    return cant

#%% Sumar 1 en cada posiciòn
def sumar_unos(lista):
    for i in range(len(lista)):
        lista[i]+=1
    return lista

#%% Intercalar
def mezclar(cadena1, cadena2):
    intercalado = ''
    i = 0
    j = 0
    while (i < len(cadena1) and j < len(cadena2)):
        intercalado += cadena1[i] + cadena2[j]
        i += 1
        j += 1
    if (j < len(cadena2)):
        intercalado += cadena2[j:]
    if (i < len(cadena1)):
        intercalado += cadena1[i:]
    return intercalado

#%% Traductor al geringoso
def geringoso(palabra):
    traduccion = ''
    for i in palabra:
        if i in 'aeiou':
            traduccion += i + 'p' + i
        else:
            traduccion += i
    return traduccion

def traductor_geringoso(lista):
    dicc = {}
    for palabra in lista:
        dicc[palabra] = geringoso(palabra)
    return dicc

#%% 
