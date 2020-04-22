# mhdflow_sr_1d

1次元特殊相対論的磁気流体シミュレーションを行うPythonスクリプトです。

# Requirements

このスクリプトを動作させるには、以下のライブラリをインストールする必要があります。

* numpy
* toml

# Basic Equations

このスクリプトでは以下の1次元特殊相対論的理想磁気流体方程式群を解いています。

* 質量保存の式(連続の式)
* 運動量保存の式
* エネルギー保存の式
* 誘導方程式

# Lear more...

詳細は数式の記述等が容易なhatena blogにて説明予定です。

# References

* 1次元特殊相対論的磁気流体方程式について詳細に記述されている論文
Mignone & Bodo, 2006
* 特殊相対論的磁気流体のHLL法についての論文  
Leismann et. al. 2005
* Primitive variablesを求める方法について詳細に書かれている論文  
Mignone & Bodo, 2006
* reconstructionについての論文  
CENO: Liu & Osher, 1997: Del Zanna & Bucciantini, 2002
MP5: Suresh & Huynh, 1997  
WENO: Liu et. al., 1994

# License
