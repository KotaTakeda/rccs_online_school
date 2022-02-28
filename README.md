# 理研データ同化オンラインスクール (基礎編) 2020

理研データ同化オンラインスクール (基礎編)は理研のデータ同化研究チームが主催となるオンラインスクールです．データ同化についての実践的な知識が身につきます．

# 内容
- 力学系とデータ同化の概念について．
- Lorenz96モデルに対するデータ同化システムの構築．
- 受講者には理解の確認のための課題が与えられています．

# コード
作成者：竹田航太

## `work.ipynb`
全体のまとめです．

## `pre_work.ipynb`
- 対応課題: 事前課題(課題1,2)
- 内容: Lorenz96の実装とカオス性の確認

## `work_data_preparation.ipynb`
- 対応課題: 課題3,4
- 内容: データ同化のためのLorenz96の真値と観測の保存．

## `work_exkf.ipynb`
- 対応課題: 課題5,6
- 内容: Extended Kalman Filter(ExKF)の実装とLorenz96モデルに対するデータ同化

## `work_3dvar.ipynb`
- 対応課題: 課題5,6
- 内容: 3次元変分法(3Dvar)の実装とLorenz96モデルに対するデータ同化

## `work_po.ipynb`
- 対応課題: 課題7
- 内容： Pertubed Observation method(PO法)によるEnsemble Kalman Filterの実装とLorenz96モデルに対するデータ同化

## `work_etkf.ipynb`
- 対応課題: 課題7
- 内容： Ensemble Transform Kalman Filter(ETKF)の実装とLorenz96モデルに対するデータ同化

## `work_letkf.ipynb`
- 対応課題: 課題7
- 内容： Local Transform Ensemble Kalman Filterの実装とLorenz96モデルに対するデータ同化


# 参考
- [Kalman Filterまとめ](https://kotatakeda.github.io/math/2020/10/07/kalman-filter.html)
- Lorenz and Emanuel, 1998: Optimal sites for supplementary weather observations:Simulation with a small model. J. Atmos. Sci., 55, 399−414.
- [Lorenz96モデルのデータ同化:準備編](https://qiita.com/litharge3141/items/41b8dd3104413529407f) - 2020/08/17
- [Lorenz96モデルのデータ同化:Extended Kalman Filter](https://qiita.com/litharge3141/items/7c1c879240d6c9d46166) - 2020/09/03
- [scipy.linalg.sqrtm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html) - 2020/09/18
- [三好](https://www.metsoc.jp/tenki/pdf/2005/2005_02_0093.pdf) - 2020/09/15
- [adaptive](http://www.data-assimilation.riken.jp/jp/events/ithes_da_2016fall/slides/160912_iTHES_Kotsuki.pdf) - 2020/09/26
- [イベント詳細](http://www.data-assimilation.riken.jp/jp/events/riken_da_tr_2020/index.html)

