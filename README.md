# pcd-watershed

path와 seed(marker)를 지정해주어야한다.

1.main.py는 ply파일을 meshlab을 통해 seed vertex index를 얻어와서 watershed(정확하게는 region growing이 알맞은 표현이다)하는 부분이다. watershed(region growing)의 prior는 color&z변위차로 구해진다. point의 neighbor는 kdtree로 구한다.<br>

2.normal_estimation.py는 하나의 cluster만 있을 때 convexhull을 통해 확실한 방향의 normal을 얻을 수 있다. 이때의 normal을 seed로 설정하여 watershed(region growing)기법을 통해 주위의 normal이 seed(확실한 normal)보다 많이 다르면(dot 부호) 방향을 역전시킨다.<br>

3.normal_polarization.py는 카메라 위치가지고 방향이 역으로 추정된 normal의 방향을 변경해주는 역할이다.(바닥은 잘되나 오목한 pcd인 경우 성능이 좋지 않다)<br>

4.watershed_normal_polaruzation(2의 옛날ver, 자동과 비자동, cluseter 제한 유무의 차이)

5.main에서 transform을 제외한 ver(사실상 1의 개선)
