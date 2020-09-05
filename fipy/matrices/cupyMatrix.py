from fipy.matrices.sparseMatrix import _SparseMatrix
import cupy as cp
from cupyx.scipy.sparse import coo_matrix, csr_matrix, dia_matrix
from fipy.tools import numerix

class _CupyMatrix(_SparseMatrix):
    
    """class wrapper for a cupy sparse matrix.
    
    _CupyMatrix is always 'NxN' and stored in the GPU
    Allows basic python operations __add__, __sub__, etc
    Facilitates matrix populating in an easy way
    Follows similar format as Scipy matrix
    """
    
    def __init__(self, matrix):
        """Create a '_CupyMatrix'.
        
        Parameters
        ----------
        matrix : cupyx.scipy.sparse.coo_matrix
            The internal Cupy-Scipy matrix
            note: Cupy matrix construction similar to that of
            _ScipyMatrix can only support COOrdination matrix
            but easily converted to CSR_matrix for arithmatic
        """
        self.matrix = matrix.tocoo()
    
    def getCoupledClass(self):
        return _CoupledCupyMeshMatrix
    
    #def __getitem__(self, index):
    #    m = self.matrix[index]
    #    if isinstance(m, type(0)) or isinstance(m, type(0.)):
    #        return m
    #    else:
    #        return _CupyMatrix(matrix=m)
    # not implemented due to unsupported indexing
    
    def nonzero(matrix):
        """The nonzero indices
        
        Return a tuple of arrays (row, col) containing the indices
        of the non-zero elements of the matrix.
        """
        A = matrix.tocoo()
        nz_mask = A.data != 0
        return (A.row[nz_mask], A.col[nz_mask])
        
        
    def _iadd(self, other, sign=1):
        """add the matrix with 'other', which can be either
        matrix or just a number
        """
        if hasattr(other, "matrix"):
            self.matrix = self.matrix + (sign * other.matrix)
        elif type(other) in [float, int]:
            fillVec = cp.repeat(other, self.matrix.nnz)
            
            self.matrix = self.matrix + coo_matrix((fillVec, nonzero(self.matrix)), self.matrix.shape)
        
        else:
            self.matrix = self.matrix + (sign * other)
        
        return self
    
    def __add__(self, other):
        """
        Add two sparse matrices should be in coo_matrix
        """
        
        if other == 0:
            return self
        else:
            return _CupyMatrix(matrix=self.matrix + other.matrix)
        # need to check if this caused lots of memory leak
        
    __radd__ = __add__
    
    def __sub__(self, other):
        """
        Subtract 
        """
        if other == 0:
            return self
        else:
            return _CupyMatrix(matrix = self.matrix - other.matrix)
        
    def __rsub__(self, other):
        return -self + other
    
    def __isub__(self, other):
        return self._iadd(other, sign=-1)
    
    def __mul__(self, other):
        """
        Multiply a sparse matrix by another sparse matrix
        # add an example of this
        
        or a sparse matrix by a vector
        # add example
        
        or a vector by a sparse matrix
        # add example
        
        """
        N = self.matrix.shape[0]
        
        if isinstance(other, _CupyMatrix):
            return _CupyMatrix(matrix=(self.matrix * other.matrix))
        else:
            shape = cupy.shape(other)
            if shape == ():
                return _CupyMatrix(matrix=(self.matrix * other))
            elif shape == (N,):
                return self.matrix * cp.array(other)
            else:
                raise TypeError
        
    def __rmul__(self, other):
        if isinstance(cp.ones(1), type(other)):
            y = self.matrix.transpose() * other
            return y
        else:
            return self * other
        
    def asformat(self, *args, **kwargs):
        # not sure what this does
        return self.matrix.asformat(*args, **kwargs)
    
    @property
    def _shape(self):
        return self.matrix.shape
    
    @property
    def _range(self):
        return list(range(self._shape[1])), list(range(self._shape[0]))
    
    def put(self, vector, id1, id2):
        """
        Put elements of 'vector' at positions of the matrix corresponding to ('id1', 'id2')
        all vector, id1, and id2 needs to be an array
        # example
        """
        assert(len(id1) == len(id2) == len(vector))
        assert(type(id1) == type(id2) == type(vector))
        
        if type(id1) == type(numerix.array([0])):
            id1 = cp.array(id1)
            id2 = cp.array(id2)
            vector = cp.array(vector)
        
        self.matrix += coo_matrix((vector, (id1, id2)), self.matrix.shape)
        
        
    def putDiagonal(self, vector):
        """
        Put elements of 'vector' along diagonal of matrix
        # example
        """
        if type(vector) in [int, float]:
            vector = cp.repeat(vector, self._shape[0])
            
        self.matrix += dia_matrix((vector, 0), shape=self.matrix.shape)
        
    #def take(self, id1, id2):
    # not implemented yet
    # cupy sparse matrix does not support indexing
    # unless it is fine to transfer the values to CPU
    
    def takeDiagonal(self):
        return self.matrix.diagonal()
    
    def addAt(self, vector, id1, id2):
        """
        Add elements of 'vector' to the positions in the matrix corresponding to ('id1', 'id2')
        # example
        """
        
        assert(len(id1) == len(id2) == len(vector))
        assert(type(id1) == type(id2) == type(vector))
        
        if type(id1) == type(numerix.array([0])):
            id1 = cp.array(id1)
            id2 = cp.array(id2)
            vector = cp.array(vector)
        
        self.matrix += coo_matrix((vector, (id1, id2)), self.matrix.shape)
    
    def addAtDiagonal(self, vector):
        if type(vector) in [type(1), type(1.)]:
            vector = cp.repeat(vector, self._shape[0])
            
        ids = cp.arange(len(vector))
        self.addAt(vector, ids, ids)
        
    @property
    def numpyArray(self):
        return self.matrix.get().toarray()
    
    #def matvec(self, x):
    #    return self * x
    # not implemented because cupy only supports PyAMGX solver
    
    def exportMmf(self, filename):
        """Export the matrix to a Matrix Market file of the given 'filename'.
        """
        from scipy.io import mmio
        mmio.mmwrite(filename, self.matrix.get())
        
    #def __getitem__(self, indices):
    # not implemented yet because cupy does not support
    # data extraction with indices
    
class _CupyMatrixFromShape(_CupyMatrix):
    
    def __init__(self, size, bandwidth=0, sizeHint=None, matrix=None):
        """
        Instantiates and wrap a Cupy Sparse Matrix
        
        Parameters
        ----------
        mesh : ~fipy.meshes.mesh.Mesh
            The `Mesh` to assemble the matrix for.
        bandwidth : int
            The proposed band width of the matrix.
        """
        if matrix is None:
            matrix = csr_matrix((size, size))
            
        _CupyMatrix.__init__(self, matrix=matrix)
        
class _CupyMeshMatrix(_CupyMatrixFromShape):
    
    def __init__(self, mesh, bandwidth=0, sizeHint=None, matrix=None, numberOfVariables=1, numberOfEquations=1):
        """Creates a `_CupyMatrixFromShape` associated with a `Mesh`.
        Parameters
        ----------
        mesh : ~fipy.meshes.mesh.Mesh
            The `Mesh` to assemble the matrix for.
        bandwidth : int
            The proposed band width of the matrix.
        numberOfVariables : int
            The columns of the matrix is determined by `numberOfVariables * self.mesh.numberOfCells`.
        numberOfEquations : int
            The rows of the matrix is determined by `numberOfEquations * self.mesh.numberOfCells`.
        """
        self.mesh = mesh
        self.numberOfVariables = numberOfVariables
        size = self.numberOfVariables * self.mesh.numberOfCells
        assert numberOfEquations == self.numberOfVariables
        _CupyMatrixFromShape.__init__(self, size=size, matrix=matrix)
        
    def __mul__(self, other):
        if isinstance(other, _CupyMeshMatrix):
            return _CupyMeshMatrix(mesh=self.mesh, matrix=(self.matrix * other.matrix))
        else:
            return _CupyMatrixFromShape.__mul__(self, other)
        
    def flush(self):
        """
        Deletes the matrix held and the GPU
        """
        if (not hasattr(self, 'cache')) or (self.cache is False):
            del self.matrix
            # add GPU memory flush
            
class _CupyIdentityMatrix(_CupyMatrixFromShape):
    """
    Represents a sparse identity matrix for Cupy
    """
    
    def __init__(self, size):
        """
        Create a sparse matrix with '1' in the diagonal
        
            >>> print(_CupyIdentityMatrix(size=3))
            
        """
        _CupyMatrixFromShape.__init__(self, size=size, bandwidth=1)
        ids = cp.arange(size)
        self.put(cp.ones(size), ids, ids)
        
class _CupyIdentityMeshMatrix(_CupyIdentityMatrix):
    def __init__(self, mesh):
        """
        Create a sparse matrix associated with 'mesh' with '1' in the diagonal
        # example
        """
        _CupyIdentityMatrix.__init__(self, size=mesh.numberOfCells)

def _test():
    import fipy.tests.doctestPlus
    return fipy.tests.doctestPlus.testmod()

if __name__ == "__main__":
    _test()