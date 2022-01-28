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
  computed: {
    plot_data() {
      return [{
        x: this.$store.state.ships.map((s) => s.pos.x),
        y: this.$store.state.ships.map((s) => s.pos.y),
        type: 'scatter'
      }]
    },
    plot_layout() {
      return {
        height: 700,
        margin: {
          t: 20
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
        shapes:
            this.$store.state.ships.map((s) => {
              return {
                type: 'path',
                path: s.pos.to_svg_path(),
                fillcolor: "red",
                line:{
                  width: 0
                },
                opacity: 0.4
              }
            }).concat([{
                type: 'path',
                path: this.$store.state.uboot_pos.to_svg_path(0.7),
                fillcolor: "black",
                line:{
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