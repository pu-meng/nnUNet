# Table 3b. SizeOV4 sampling control

<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th colspan="3">LiTS</th>
      <th colspan="3">3D-IRCADb-01</th>
      <th colspan="3">HCC-TACE-Seg</th>
    </tr>
    <tr>
      <th>Avg. Dice</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>Avg. Dice</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>Avg. Dice</th>
      <th>Recall</th>
      <th>Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MedNeXt †</td>
      <td align="right">85.61</td>
      <td align="right">73.34</td>
      <td align="right">84.81</td>
      <td align="right">82.80</td>
      <td align="right"><strong>65.54</strong></td>
      <td align="right">78.77</td>
      <td align="right">62.79</td>
      <td align="right">35.82</td>
      <td align="right">62.95</td>
    </tr>
    <tr>
      <td>MedNeXt + SizeOV4 †</td>
      <td align="right"><strong>85.90</strong></td>
      <td align="right">73.61</td>
      <td align="right"><strong>85.42</strong></td>
      <td align="right">83.91</td>
      <td align="right">65.00</td>
      <td align="right">81.54</td>
      <td align="right">57.75</td>
      <td align="right">27.82</td>
      <td align="right">61.37</td>
    </tr>
    <tr>
      <td>Ours: + MLA + MoE</td>
      <td align="right">85.03</td>
      <td align="right"><strong>74.01</strong></td>
      <td align="right">82.05</td>
      <td align="right">81.74</td>
      <td align="right">63.35</td>
      <td align="right">77.46</td>
      <td align="right"><strong>64.58</strong></td>
      <td align="right"><strong>39.51</strong></td>
      <td align="right">66.49</td>
    </tr>
    <tr>
      <td>Ours + SizeOV4</td>
      <td align="right">85.61</td>
      <td align="right">73.09</td>
      <td align="right">81.43</td>
      <td align="right"><strong>84.24</strong></td>
      <td align="right">64.61</td>
      <td align="right"><strong>83.62</strong></td>
      <td align="right">64.48</td>
      <td align="right">36.53</td>
      <td align="right"><strong>68.71</strong></td>
    </tr>
  </tbody>
</table>

本表在普通 MedNeXt 和主方法上分别加入 SizeOV4，用于区分结构贡献与采样策略影响。SizeOV4 对 MedNeXt 的 LiTS/IRCADb/HCC Avg. Dice 变化为 `+0.29/+1.11/-5.03` 个百分点；对 Ours 的变化为 `+0.58/+2.50/-0.10` 个百分点。这表明 SizeOV4 的作用依赖网络结构和数据域，不支持“SizeOV4 在三域稳定增益”的结论。

† `MedNeXt` 和 `MedNeXt + SizeOV4` 的 LiTS 指标可由现存 `summary.json` 复核，且报告与 PNG 已保留；但预期 26 例的预测 NIfTI 现均为 0/26，按当前交付规则标记为“部分完成”，不等同于完整产物交付。
