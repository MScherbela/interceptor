<template>
  <div class="mt-2 mb-2">
    <b-form @submit.prevent="$store.commit('add_ship', $data)">
      <b-form-row>
        <b-col>
          <b-form-group label="Ship name" label-for="ship_name">
            <b-form-input v-model="name" id="ship_name" type="text" required/>
          </b-form-group>
        </b-col>
        <b-col>
          <b-form-group label-for="ship_type" label="Ship type">
            <b-form-select v-model="ship_type" id="ship_type" :options="ship_type_options"/>
          </b-form-group>
        </b-col>
      </b-form-row>
      <b-form-row>
        <b-col>
          <b-form-group label="BRT" label-for="ship_brt">
            <b-form-input v-model="tons" id="ship_brt" type="number"/>
          </b-form-group>
        </b-col>
        <b-col>
          <b-form-group label="Speed / knts" label-for="ship_speed">
            <b-form-input v-model="speed" id="ship_speed" type="number" step="0.1"/>
          </b-form-group>
        </b-col>
      </b-form-row>
      <b-form-row>
        <b-col>
          <b-form-group label="Distanz" label-for="ship_distance">
            <b-form-input v-model="distance" id="ship_distance" type="number" step="0.01"/>
          </b-form-group>
        </b-col>
        <b-col>
          <b-form-group label="Kurs" label-for="ship_heading">
            <b-form-input v-model="heading" id="ship_heading" type="number"/>
          </b-form-group>
        </b-col>
        <b-col>
          <b-form-group label="Peilung" label-for="ship_rel_direction">
            <b-form-input v-model="rel_direction" id="ship_rel_direction" type="number"/>
          </b-form-group>
        </b-col>
      </b-form-row>

      <b-form-row>
        <b-col>
        <b-button type="submit" variant="primary">Add Ship</b-button>
          </b-col>
      </b-form-row>
    </b-form>
  </div>
</template>

<script>
export default {
  name: "AddShipForm",
  props: {
    ship_type_options: {
      type: Array,
      default: function () {
        return ["Unbekannt", "Zerstörer", "Handelsschiff"]
      }
    }
  },
  data() {
    return {
      name: "",
      ship_type: "Unbekannt",
      tons: 1000,
      speed: 7.0,
      distance: 5.0,
      heading: 20,
      rel_direction: 90
    }
  }
}
</script>

<style scoped>

</style>