# Squid Control

The Squid Control software is a Python package that provides a simple interface to control the Squid microscope. The software is designed to be used with the Squid microscope (made by Cephla Inc.).

## Installation and Usage

See the [installation guide](./docs/installation.md) for instructions on how to install and use the software.

### Usage

To run the software, use the following command:
```
python -m squid_control --config HCS_v2
```

If you want to use a different configuration file, you can specify the path to the configuration file:
```
python -m squid_control --config /home/user/configuration_HCS_v2.ini
```

To start simulation mode, use the following command:
```
python -m squid_control --config HCS_v2 --simulation
```

## About

<img style="width:60px;" src="./docs/assets/cephla_logo.svg"> Cephla Inc. 



