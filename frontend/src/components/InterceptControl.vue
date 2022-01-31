<template>
  <b-card>
    <form v-on:submit.prevent="get_intercept_course">
      <b-form-row>
                <b-col cols="8">
          <b-form-group label-for="target_ship" label="Ziel">
            <b-form-select v-model="target_id" id="target_ship" :options="target_options"/>
          </b-form-group>
        </b-col>
        <b-col>
          <b-form-group label="Segmente" label-for="n_segments">
            <b-form-input id="n_segments" v-model="n_segments" type="number" step="1" max="5" min="1"/>
          </b-form-group>
        </b-col>
      </b-form-row>
      <b-form-row>
        <b-col>
          <b-button type="submit">Abfangen</b-button>
        </b-col>
      </b-form-row>
    </form>
    <div v-if="intercept != null">
      <ul v-for="wp in intercept.route" :key="wp.t">
        <li><b>{{wp.timestamp}}:</b> Kurs {{wp.heading.toFixed(1)}} deg, {{wp.target_dist.toFixed(1)}} sm</li>
      </ul>
    </div>
  </b-card>
</template>


<script>
import axios from 'axios';

export default {
  name: "InterceptControl",
  data() {
    return {
      n_segments: 2,
      target_id: 0,
      // backend_url: "http://localhost:5000/api",
      backend_url: "https://uboot.scherbela.com/api"
    }
  },
  props: {
    ships: Array,
    uboot_pos: Object,
    time: Number,
    intercept: Object
  },
  computed: {
    target_options() {
      return this.ships.map((s) => {
        return {value: s.id, text: s.name}
      })
    }
  },
  methods: {
    get_intercept_params() {
      const target = this.ships.find(s => s.id == this.target_id).pos
      return {
        x: this.uboot_pos.x,
        y: this.uboot_pos.y,
        heading: this.uboot_pos.heading,
        speed: this.uboot_pos.speed,
        target_x: target.x,
        target_y: target.y,
        target_heading: target.heading,
        target_speed: target.speed,
        time: this.time,
        n_segments: this.n_segments
      }
    },
    get_intercept_course() {
      axios.get(this.backend_url,
          {params: this.get_intercept_params()}
      ).then(response => {
            console.log(response.data);
            this.$store.commit('set_intercept', response.data)
          }
      ).catch(e => console.log(e))
    }
  }
}
</script>

<style scoped>

</style>