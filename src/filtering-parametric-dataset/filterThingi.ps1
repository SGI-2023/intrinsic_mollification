# we will run the executable "\gc-polyscope-project-template\build\bin\Release\gc_project.exe" with the mesh path as an argument
# the executable will return 0 or 1 depending on whether the mesh is valid or not
# if the mesh is valid, we will add its path to the list of valid mesh paths
# if the mesh is invalid, we will add its path to the list of invalid mesh paths
# we will then write the list of valid mesh paths to a file called "validMeshPaths.txt"
# we will then write the list of invalid mesh paths to a file called "invalidMeshPaths.txt"
# This is powershell code, not python code

$validMeshPaths = @()
$invalidMeshPaths = @()
$tooLargeMeshPaths = @()
$tooNiceMeshPaths = @()

# get the path to the directory containing the meshes
$meshesDir = "./Obj_Files/Obj_Files/Artist_UVs/Uncut/"

# get the path to the directory containing the executable
$executableDir = "./build/bin/gc_project"

$i = 0
# run the executable on each mesh file in the meshes directory
foreach ($mesh in (Get-ChildItem $meshesDir)) {
    $meshPath = $mesh
    # get the return value of the executable if the mesh file size is < 20 MB
    $output = ""
    if ((Get-Item $meshPath).length -gt 20000000) {
        $tooLargeMeshPaths += $meshPath
        continue
    }
    else
    {
        $output = & $executableDir $meshPath
    }

    # if the output is empty, then the executable failed to run, so such meshes will be added to the invalid meshes list, get the length of the output
    if ($output.length -eq 0) {
        $invalidMeshPaths += $meshPath
        continue
    }

    # first character of the output should be 0 or 1
    if ($output[0] -eq "0") {
        $validMeshPaths += $meshPath
    } elseif ($output[0] -eq "1") {
        $invalidMeshPaths += $meshPath
    }
    elseif ($output[0] -eq "2") {
        $tooNiceMeshPaths += $meshPath
    }

    # increment the counter
    $i++

    if ($i % 200 -eq 0) {
        Write-Host $i
        # print the count of valid and invalid meshes
        Write-Host "Valid meshes: " $validMeshPaths.Count
        Write-Host "Invalid meshes: " $invalidMeshPaths.Count
        Write-Host "Too large meshes: " $tooLargeMeshPaths.Count
        Write-Host "Too nice meshes: " $tooNiceMeshPaths.Count
    }
}
# print the count of valid and invalid meshes
Write-Host "Valid meshes: " $validMeshPaths.Count
Write-Host "Invalid meshes: " $invalidMeshPaths.Count
Write-Host "Too large meshes: " $tooLargeMeshPaths.Count
Write-Host "Too nice meshes: " $tooNiceMeshPaths.Count

# write the list of valid mesh paths to a file
$validMeshPaths | Out-File "validMeshPaths.txt"

# write the list of invalid mesh paths to a file
$invalidMeshPaths | Out-File "invalidMeshPaths.txt"

# write the list of too large mesh paths to a file
$tooLargeMeshPaths | Out-File "tooLargeMeshPaths.txt"

# write the list of too nice mesh paths to a file
$tooNiceMeshPaths | Out-File "tooNiceMeshPaths.txt"



