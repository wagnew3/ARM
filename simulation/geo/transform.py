import numpy as np
import scipy.linalg

from pyquaternion import Quaternion


class Transform:
    """
    Represents an affine transformation composed of translation, rotation, scale
    """
    def __init__(self, translation=np.asarray((0, 0, 0)), rotation=Quaternion(), scale=np.asarray((1, 1, 1))):
        self._t = translation
        self._r = rotation
        self._s = scale

    @property
    def rotation(self):
        return self._r

    @property
    def translation(self):
        return self._t

    @property
    def scale(self):
        return self._s

    @classmethod
    def from_mat4(cls, m):
        """
        Construct Transform from matrix m
        :param m: 4x4 affine transformation matrix
        :return: rotation (quaternion), translation, scale
        """
        translation = m[:3, 3]
        scale_rotation = m[:3, :3]
        u, _ = scipy.linalg.polar(scale_rotation)
        rotation_q = Quaternion(matrix=u)
        r_axis = rotation_q.get_axis(undefined=[0, 1, 0])
        # ensure that rotation axis is +y
        if np.array(r_axis).dot(np.array([0, 1, 0])) < 0:
            rotation_q = -rotation_q
        scale = np.linalg.norm(scale_rotation, axis=0)
        return cls(translation=translation, rotation=rotation_q, scale=scale)

    @classmethod
    def from_mat4x4_flat_row_major(cls, mat):
        return cls.from_mat4(np.array(mat).reshape((4, 4)).transpose())

    @classmethod
    def from_node(cls, node):
        if hasattr(node, 'transform'):
            # get rotation and translation out of 4x4 xform
            return cls.from_mat4x4_flat_row_major(node.transform)
        elif hasattr(node, 'type') and hasattr(node, 'bbox'):
            p_min = np.array(node.bbox['min'])
            p_max = np.array(node.bbox['max'])
            center = (p_max + p_min) * 0.5
            half_dims = (p_max - p_min) * 0.5
            return cls(translation=center, scale=half_dims)
        else:
            return None

    def as_mat4(self):
        m = np.identity(4)
        m[:3, 3] = self._t
        r = self._r.rotation_matrix
        for i in range(3):
            m[:3, i] = r[:, i] * self._s[i]
        return m

    def as_mat4_flat_row_major(self):
        return list(self.as_mat4().transpose().flatten())

    def inverse(self):
        return self.from_mat4(np.linalg.inv(self.as_mat4()))

    def set_translation(self, t):
        self._t = t

    def translate(self, t):
        self._t += t

    def rotate(self, radians, axis):
        q = Quaternion(axis=axis, radians=radians)
        self._r = q * Quaternion(matrix=self._r)

    def set_rotation(self, radians, axis=np.asarray((0, 1, 0))):
        self._r = Quaternion(axis=axis, radians=radians)

    def rotate_y(self, radians):
        self.rotate(radians, [0, 1, 0])

    def rescale(self, s):
        self._s = np.multiply(self._s, s)

    def transform_point(self, p):
        # TODO cache optimization
        return np.matmul(self.as_mat4(), np.append(p, [1], axis=0))[:3]

    def transform_direction(self, d):
        # TODO cache optimization
        return np.matmul(np.transpose(self._r.rotation_matrix), d)
