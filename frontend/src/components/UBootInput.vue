<template>
  <div class="my-1">
    <b-card class="cardbackground" body-class="compactcardbody">
      <b-card-title class="my-1">
        <h4 class="frakturfont">U-Boot Kurs</h4>
      </b-card-title>
      <form>
        <b-form-row>
          <b-col>
            <b-form-group label="Kurs" label-for="heading">
              <b-form-input id="heading" v-model="heading" type="number" step="1" max="359" min="0"/>
            </b-form-group>
          </b-col>
          <b-col>
            <b-form-group label="kn" label-for="speed">
              <b-form-input id="speed" v-model="speed" type="number" step="1" max="359" min="0"/>
            </b-form-group>
          </b-col>
        </b-form-row>
      </form>
    </b-card>
  </div>
</template>

<script>
export default {
  name: "UBootInput",
  computed: {
    heading: {
      get() {
        return this.$store.state.uboot_pos.heading
      },
      set(heading) {
        return this.$store.commit('update_uboot_heading', parseFloat(heading) || 0)
      }
    },
    speed: {
      get() {
        return this.$store.state.uboot_pos.speed
      },
      set(speed) {
        return this.$store.commit('update_uboot_speed', parseFloat(speed) || 0)
      }
    }
  }
}
</script>

<style scoped>

</style>