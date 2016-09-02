#!/usr/bin/env nix-shell

with import <nixpkgs> {};

let
    envname = "py27-ve";
    python = python27Full;
    pyp = pkgs.python27Packages;
    activate = ''
    export NIX_ENV="[${envname}] "
    if ! test -d py27-venv; then
      virtualenv py27-venv
      source py27-venv/bin/activate
      pip install imageio
    fi
    source py27-venv/bin/activate
    '';
in

buildPythonPackage { 
  name = "${envname}-env";
  buildInputs = [
     pyp.virtualenv
     python
    ];
   pythonPath = with pyp; [
     virtualenv
     setuptools
     readline
     seaborn
     ipython
     mahotas
     sqlite3
     Theano
     pandas
     numpy
  ];
  src = null;
  shellHook = activate;
  extraCmds = activate;
}

