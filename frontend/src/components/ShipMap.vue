<template>
<!--/*  <b-card style="height: 95%" ref="shipmap_div" class="cardbackground mapcard">*/-->
    <Plotly :data="plot_data" :layout="plot_layout" :displayModeBar="false" class="mb-3"/>
<!--  </b-card>-->
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
    uboot_pos: Object,
    intercept: Object
  },
  methods: {
    get_ship_traces() {
      return this.ships.map((s) => {
        const rel_pos = this.uboot_pos.get_relative_position(s.pos)
        return {
          x: [this.uboot_pos.x, s.pos.x],
          y: [this.uboot_pos.y, s.pos.y],
          type: 'scatter',
          name: s.name + ": " + rel_pos.distance.toFixed(1) + " sm, " + rel_pos.rel_direction.toFixed(0) + "°",
          showlegend: true,
          line: {
            color: s.color
          }
        }
      })
    },
    get_traces_to_intercepted_ships() {
      return this.intercept.final_ship_positions.map(s => {
        const rel_pos = this.intercept.final_uboot_pos.get_relative_position(s.pos)
        return {
          x: [this.intercept.final_uboot_pos.x, s.pos.x],
          y: [this.intercept.final_uboot_pos.y, s.pos.y],
          type: 'scatter',
          name: s.name + ": " + rel_pos.distance.toFixed(1) + " sm, " + rel_pos.rel_direction.toFixed(0) + "°",
          opacity: 0.6,
          showlegend: true,
          line: {
            color: s.color,
            dash: 'dash'
          }
        }
      })
    },
    get_intercept_trace() {
      const route = this.intercept.route
      return {
        x: route.map(wp => wp.x),
        y: route.map(wp => wp.y),
        type: 'scatter',
        name: 'Intercept',
        line: {
          color: 'black',
        },
        showlegend: true
      }
    },
    get_ship_shape(pos, color, size, opacity = 0.4) {
      return {
        type: 'path',
        path: pos.to_svg_path(size),
        fillcolor: color,
        line: {
          width: 0
        },
        opacity: opacity,
      }
    },
    get_uboot_shape() {
      return this.get_ship_shape(this.uboot_pos, 'black', 0.7, 0.6)
    },
    adjust_plot_height() {
      // this.plot_height = this.$refs.shipmap_div.clientHeight - 50
      this.plot_height = window.innerHeight * 0.85
    }
  },
  data() {
    return {
      plot_height: 500
    }
  },
  mounted() {
    window.addEventListener("resize", this.adjust_plot_height)
    this.adjust_plot_height()
  },
  computed: {
    plot_data() {
      let traces = this.get_ship_traces()
      if (this.intercept != null) {
        traces = traces.concat(this.get_traces_to_intercepted_ships())
        traces.push(this.get_intercept_trace())
      }
      return traces
    },
    plot_layout() {
      return {
        height: this.plot_height,
        autosize: true,
        images: [
          {
            source: "paper.jpg",
            // source: "https://raw.githubusercontent.com/cldougl/plot_images/add_r_img/vox.png",
            x: 0,
            y: 1,
            sizex: 1,
            sizey: 1,
            xref: "paper",
            yref: "paper",
            sizing: "fill",
            layer: "below"
          }
        ],
        margin: {
          t: 0,
          b: 0,
          l: 0,
          r: 0
        },
        yaxis: {
          // range: [-20, 20],
          dtick: 1,
          zeroline: false,
          showticklabels: false
        },
        xaxis: {
          // range: [-20, 20],
          scaleanchor: 'y',
          scaleratio: 1,
          dtick: 1,
          zeroline: false,
          showticklabels: false
        },
        legend: {
          x: 0.02,
          y: 0.02,
          bgcolor: 'rgba(1.0, 1.0, 1.0, 0.7)'
        },
        shapes:
            this.ships.map(s => this.get_ship_shape(s.pos, s.color, 0.5, 0.7)
            ).concat(this.intercept != null ? this.intercept.final_ship_positions.map(p => this.get_ship_shape(p.pos, p.color, 0.5, 0.3)) : []
            ).concat([this.get_uboot_shape()])
      }
    }
  },
}
</script>

<style scoped>

.mapcard {
  /*height: max-content;*/
  overflow: hidden;
}

</style>