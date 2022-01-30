<template>
  <div>
    <Plotly :data="plot_data" :layout="plot_layout" :displayModeBar="false"/>
  </div>
</template>


<script>
import {Plotly} from 'vue-plotly'

export default {
  name: "ShipMap",
  components: {
    Plotly,
  },
  props: {
    ships: Array,
    uboot_pos: Object
  },
  computed: {
    plot_data() {
      return this.ships.map((s) => {
        const rel_pos = this.uboot_pos.get_relative_position(s.pos)
        return {
          x: [this.uboot_pos.x, s.pos.x],
          y: [this.uboot_pos.y, s.pos.y],
          type: 'scatter',
          name: s.name + ": " + rel_pos.distance.toFixed(1) + " sm, " + rel_pos.rel_direction.toFixed(0) + "°",
          showlegend: true
        }
      })
    },
    plot_layout() {
      return {
        height: 700,
        margin: {
          t: 20,
          b: 20,
          l: 20,
          r: 20
        },
        yaxis: {
          range: [-10, 10],
          scaleanchor: 'x',
          scaleratio: 1,
          dtick: 1,
          zeroline: false
        },
        xaxis: {
          range: [-10, 10],
          dtick: 1,
          zeroline: false
        },
        legend:{
          x: 0.02,
          y: 0.02
        },
        shapes:
            this.ships.map((s) => {
              return {
                type: 'path',
                path: s.pos.to_svg_path(),
                fillcolor: s.is_warship() ? "red" : "blue",
                line: {
                  width: 0
                },
                opacity: 0.4
              }
            }).concat([{
              type: 'path',
              path: this.uboot_pos.to_svg_path(0.7),
              fillcolor: "black",
              line: {
                width: 0
              },
              opacity: 0.4
            }])
      }
    }
  },
}
</script>

<style scoped>

</style>