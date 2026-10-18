{
  description = "A development environment Python";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        packages.default = pkgs.python3Packages.buildPythonPackage {
          pname = "zubar-sim";
          version = "0.1.0";
          src = ./.;

          pyproject = true;

          nativeBuildInputs = [
            pkgs.wrapGAppsHook3
          ];

          build-system = with pkgs.python3Packages; [
            hatchling
          ];

          propagatedBuildInputs = with pkgs.python3Packages; [
            numpy
            matplotlib
            (opencv4.override { enableGtk3 = true; })
          ];

          pythonRemoveDeps = [
            "opencv-contrib-python"
          ];

          doCheck = false;
        };
      }
    );
}
