import os
import sys
import subprocess

from setuptools import Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.core import setup

__CMAKE_PREFIX_PATH__       =   None
__ENVIRONMENT_PATH__        =   os.path.dirname(os.path.realpath(__file__)) + "/gym_ignition/env/env/panda"
__DEBUG__                   =   None
__ENVIRONMENT_BUILD_NAME__  =   "panda"

if "--CMAKE_PREFIX_PATH" in sys.argv:
    index = sys.argv.index('--CMAKE_PREFIX_PATH')
    __CMAKE_PREFIX_PATH__ = sys.argv[index + 1]
    sys.argv.remove("--CMAKE_PREFIX_PATH")
    sys.argv.remove(__CMAKE_PREFIX_PATH__)

for i in ["--Debug", "--DEBUG"]:
    if i in sys.argv:
        index = sys.argv.index(i)
        sys.argv.remove(i)
        __DEBUG__ = True

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " + ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        if __CMAKE_PREFIX_PATH__ is not None:
            cmake_args.append('-DCMAKE_PREFIX_PATH=' + __CMAKE_PREFIX_PATH__)

        if __ENVIRONMENT_BUILD_NAME__ is not None:
            cmake_args.append('-DENVIRONMENT_BUILD_NAME=' + __ENVIRONMENT_BUILD_NAME__)

        cmake_args.append('-DGYM_IGN_ENVIRONMENT_INCLUDE_PATH=' + __ENVIRONMENT_PATH__)

        cfg = 'Debug' if __DEBUG__ else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j2']

        cmake_args.append('-Wall')
        build_args.append('VERBOSE=1')
        build_args.append('-Wall')

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())

        # create a build/ directory and launch the CMake build process automatically
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='gym_ignition',
    version='0.0.0',
    author='Nico Palomo',
    license="GNU",
    packages=find_packages(),
    author_email='npalomo@student.ethz.ch',
    description='gym for ignition',
    long_description='',
    ext_modules=[CMakeExtension('_gym_ignition')],
    install_requires=['gym>=0.2.3', 'ruamel.yaml', 'numpy', 'stable_baselines==2.8'],
    cmdclass=dict(build_ext=CMakeBuild),
    include_package_data=True,
    zip_safe=False,
)

# To build the code:
# python3 setup.py install --CMAKE_PREFIX_PATH $IGN_GAZEBO_SYSTEM_PLUGIN_PATH