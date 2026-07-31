# Complex pytrajectory CLI commands, kept here so you can copy one instead of
# retyping it from the README. Each opens an interactive window - run one at
# a time (pass its number), rather than executing the whole file.
#
# Usage (from the repo root):
#   .\examples\cli_showcase.ps1 1
#   .\examples\cli_showcase.ps1 2

param(
    [Parameter(Position = 0)]
    [string]$Example
)

function Engagement-FixedBox {
    # Friend + enemy aircraft and a missile, each with their own model and
    # color, animated with a growing trail, camera box pinned to a fixed size.
    pytrajectory `
        --traj examples\trajectory_friend.mat color=green model=models\f-16.mat `
        --traj examples\trajectory_enemy.mat  color=black model=models\f-16.mat `
        --traj examples\trajectory_aam.mat    color=red   model=models\missile.mat scale=10 `
        --line-width 100 --animate --trail `
        --xlim -3000 1000 --zlim 0 1000
}

function Engagement-AutoBox {
    # Same scene, but the camera box auto-fits to the data instead of being
    # pinned - a bigger, uncropped view.
    pytrajectory `
        --traj examples\trajectory_friend.mat color=green model=models\f-16.mat `
        --traj examples\trajectory_enemy.mat  color=black model=models\f-16.mat `
        --traj examples\trajectory_aam.mat    color=red   model=models\missile.mat scale=10 `
        --line-width 100 --animate --trail
}

switch ($Example) {
    "1" { Engagement-FixedBox }
    "2" { Engagement-AutoBox }
    default {
        Write-Host "Usage: .\cli_showcase.ps1 <example-number>"
        Write-Host ""
        Write-Host "  1  friend/enemy/missile engagement, fixed camera box"
        Write-Host "  2  same engagement, auto-fit camera box"
        exit 1
    }
}
